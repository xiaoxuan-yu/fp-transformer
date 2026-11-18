"""Prepare the PoseBusters journal subset for torsion-only Vina sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors



from examples.vina.manifest_io import write_manifest_csv


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "data/raw/posebusters_benchmark_set"
DEFAULT_OUTPUT = HERE / "data/prepared"
DEFAULT_MANIFEST = HERE / "manifest.json"
DEFAULT_SUBSET = HERE / "paper_subset_308_ids.txt"
SOURCE_ARCHIVE_MD5 = "f004ac7c4e68317b5348497d2bb6bee6"
SUBSET_SHA256 = "a69a7b6b9a5a52531933078ef983e6c069e3a987a1d7a733bd7d72cbe1793de6"
ALLOWED_ELEMENTS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si"}
ALLOWED_PDBQT_TYPES = {
    "H",
    "HD",
    "A",
    "C",
    "N",
    "NA",
    "O",
    "OA",
    "S",
    "SA",
    "P",
    "F",
    "Cl",
    "CL",
    "Br",
    "BR",
    "I",
    "Si",
    "At",
    "Mg",
    "Mn",
    "Zn",
    "Ca",
    "Fe",
}
WATER_RESIDUES = {"HOH", "WAT", "DOD"}
REMOVABLE_POCKET_HETERO_RESIDUES = {
    "ACT",
    "BOG",
    "DMS",
    "EDO",
    "GOL",
    "K",
    "NA",
    "PEG",
    "PEO",
    "PO4",
    "SO4",
    "TRS",
}
SUPPORTED_MONOATOMIC_ION_TYPES = {
    "CA": "Ca",
    "CL": "Cl",
    "FE": "Fe",
    "FE2": "Fe",
    "IOD": "I",
    "MG": "Mg",
    "MN": "Mn",
    "ZN": "Zn",
}
STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
POCKET_RADIUS = 10.0
SYSTEM_ID_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]+$")


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_subset_ids(path: Path) -> list[str]:
    observed = sha256sum(path)
    if observed != SUBSET_SHA256:
        raise RuntimeError(
            f"journal subset SHA256 mismatch: expected {SUBSET_SHA256}, "
            f"got {observed}"
        )
    system_ids = path.read_text().split()
    if len(system_ids) != 308 or len(set(system_ids)) != 308:
        raise ValueError("journal subset must contain 308 unique system IDs")
    invalid = [item for item in system_ids if not SYSTEM_ID_PATTERN.fullmatch(item)]
    if invalid:
        raise ValueError(f"invalid journal-subset system IDs: {invalid[:5]}")
    return system_ids


def _load_ligand(path: Path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecules = [molecule for molecule in supplier if molecule is not None]
    if len(molecules) != 1:
        raise ValueError(f"expected one valid molecule, found {len(molecules)}")
    return molecules[0]


def _atom_coordinate(atom_line: str) -> tuple[float, float, float]:
    return (
        float(atom_line[30:38]),
        float(atom_line[38:46]),
        float(atom_line[46:54]),
    )


def _nearby_hetero_residues(
    protein_path: Path, ligand_coordinates
) -> list[str]:
    residues = set()
    for line in protein_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        residue = line[17:20].strip()
        if residue in STANDARD_AMINO_ACIDS or residue in WATER_RESIDUES:
            continue
        coordinate = _atom_coordinate(line)
        if any(
            math.dist(coordinate, ligand_coordinate) <= POCKET_RADIUS
            for ligand_coordinate in ligand_coordinates
        ):
            chain = line[21].strip() or "_"
            number = line[22:26].strip()
            residues.add(f"{residue}:{chain}:{number}")
    return sorted(residues)


def _write_clean_receptor(source: Path, destination: Path) -> None:
    lines = []
    for line in source.read_text().splitlines():
        if line.startswith("TER"):
            lines.append(line)
        elif line.startswith(("ATOM  ", "HETATM")):
            if line[17:20].strip() in STANDARD_AMINO_ACIDS:
                lines.append(line)
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in lines):
        raise ValueError("clean receptor contains no atoms")
    destination.write_text("\n".join(lines + ["END"]) + "\n")


def _append_supported_ions(source: Path, receptor_pdbqt: Path) -> int:
    """Append supported single-atom ions to Meeko's rigid receptor PDBQT."""

    pdbqt_lines = receptor_pdbqt.read_text().splitlines()
    serials = [
        int(line.split()[1])
        for line in pdbqt_lines
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    serial = max(serials, default=0)
    appended = []
    for line in source.read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        residue = line[17:20].strip()
        atom_type = SUPPORTED_MONOATOMIC_ION_TYPES.get(residue)
        if atom_type is None:
            continue
        serial += 1
        name = line[12:16].strip() or residue
        chain = line[21].strip() or "_"
        number = line[22:26].strip() or "0"
        x, y, z = _atom_coordinate(line)
        appended.append(
            # Use the whitespace-delimited receptor convention consumed by
            # parse_pdbqt_atoms(); standard PDB's HETATM+serial field has no
            # delimiter and would make the ion name look like the serial.
            f"ATOM  {serial:5d} {name:^4s} {residue:>3s} {chain:1s}"
            f"{number:>4s}    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00     0.000 {atom_type}"
        )
    receptor_pdbqt.write_text("\n".join(pdbqt_lines + appended) + "\n")
    return len(appended)


def _write_hydrogenated_ligand(molecule, destination: Path) -> None:
    molecule = Chem.AddHs(molecule, addCoords=True)
    writer = Chem.SDWriter(str(destination))
    writer.write(molecule)
    writer.close()


def _run(command: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    message = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    return completed.returncode == 0, message[-4000:]


def _pdbqt_summary(path: Path) -> tuple[int, int, list[str]]:
    atoms = 0
    torsions = 0
    unsupported = set()
    for line in path.read_text().splitlines():
        if line.startswith("BRANCH"):
            torsions += 1
        elif line.startswith(("ATOM  ", "HETATM")):
            atoms += 1
            atom_type = line.split()[-1]
            if atom_type not in ALLOWED_PDBQT_TYPES:
                unsupported.add(atom_type)
    return atoms, torsions, sorted(unsupported)


def _base_record(system_id: str, source_dir: Path, in_subset: bool) -> dict:
    pdb_id, ligand_code = system_id.split("_", maxsplit=1)
    return {
        "system_id": system_id,
        "pdb_id": pdb_id,
        "ligand_code": ligand_code,
        "journal_subset": in_subset,
        "source_ligand": str(
            (source_dir / f"{system_id}_ligand.sdf").relative_to(HERE)
        ),
        "source_protein": str(
            (source_dir / f"{system_id}_protein.pdb").relative_to(HERE)
        ),
        "status": "pending" if in_subset else "excluded",
        "reason": None if in_subset else "not_in_journal_subset",
        "heavy_atoms": None,
        "rdkit_rotatable_bonds": None,
        "pdbqt_torsions": None,
        "max_ring_size": None,
        "pocket_hetero_residues": [],
        "removed_pocket_hetero_residues": [],
        "retained_pocket_ions": [],
        "blocking_pocket_hetero_residues": [],
        "retained_receptor_ion_atoms": None,
        "receptor_pdbqt_atoms": None,
        "ligand_pdbqt_atoms": None,
        "primary_benchmark": False,
    }


def prepare_system(
    source_dir: Path,
    output_dir: Path,
    ligand_cli: Path,
    receptor_cli: Path,
    *,
    in_subset: bool,
) -> dict:
    system_id = source_dir.name
    record = _base_record(system_id, source_dir, in_subset)
    if not in_subset:
        return record

    ligand_path = source_dir / f"{system_id}_ligand.sdf"
    protein_path = source_dir / f"{system_id}_protein.pdb"
    if not ligand_path.exists() or not protein_path.exists():
        record.update(status="filtered", reason="missing_source_file")
        return record
    try:
        molecule = _load_ligand(ligand_path)
    except Exception as error:  # RDKit reports heterogeneous parse failures
        record.update(status="filtered", reason=f"ligand_parse:{error}")
        return record

    elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    rings = molecule.GetRingInfo().AtomRings()
    max_ring_size = max((len(ring) for ring in rings), default=0)
    record.update(
        heavy_atoms=molecule.GetNumHeavyAtoms(),
        rdkit_rotatable_bonds=int(
            rdMolDescriptors.CalcNumRotatableBonds(molecule)
        ),
        max_ring_size=max_ring_size,
    )
    if len(Chem.GetMolFrags(molecule)) != 1:
        record.update(status="filtered", reason="disconnected_ligand")
        return record
    unsupported_elements = sorted(elements - ALLOWED_ELEMENTS)
    if unsupported_elements:
        record.update(
            status="filtered",
            reason="unsupported_elements:" + ",".join(unsupported_elements),
        )
        return record
    if max_ring_size >= 7:
        record.update(status="filtered", reason=f"macrocycle:ring_size={max_ring_size}")
        return record

    conformer = molecule.GetConformer()
    ligand_coordinates = [
        tuple(conformer.GetAtomPosition(index))
        for index in range(molecule.GetNumAtoms())
        if molecule.GetAtomWithIdx(index).GetAtomicNum() > 1
    ]
    pocket_hetero = _nearby_hetero_residues(protein_path, ligand_coordinates)
    target_ligand_code = system_id.split("_", maxsplit=1)[1]
    removable_names = REMOVABLE_POCKET_HETERO_RESIDUES | {target_ligand_code}
    removable = [
        residue
        for residue in pocket_hetero
        if residue.split(":", maxsplit=1)[0]
        in removable_names
    ]
    retained_ions = [
        residue
        for residue in pocket_hetero
        if residue.split(":", maxsplit=1)[0]
        in SUPPORTED_MONOATOMIC_ION_TYPES
    ]
    blocking = [
        residue
        for residue in pocket_hetero
        if residue not in removable and residue not in retained_ions
    ]
    record.update(
        pocket_hetero_residues=pocket_hetero,
        removed_pocket_hetero_residues=removable,
        retained_pocket_ions=retained_ions,
        blocking_pocket_hetero_residues=blocking,
    )
    if blocking:
        record.update(status="filtered", reason="pocket_hetero:" + ",".join(blocking))
        return record

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"posebusters_{system_id}_", dir=output_dir
    ) as temporary_name:
        temporary = Path(temporary_name)
        clean_receptor = temporary / "receptor.pdb"
        ligand_h = temporary / "ligand_h.sdf"
        receptor_pdbqt = temporary / "receptor.pdbqt"
        ligand_pdbqt = temporary / "ligand.pdbqt"
        try:
            _write_clean_receptor(protein_path, clean_receptor)
            _write_hydrogenated_ligand(molecule, ligand_h)
        except Exception as error:
            record.update(status="filtered", reason=f"input_cleanup:{error}")
            return record

        ligand_ok, ligand_log = _run(
            [
                str(ligand_cli),
                "-i",
                str(ligand_h),
                "-o",
                str(ligand_pdbqt),
                "--charge_model",
                "zero",
            ]
        )
        if not ligand_ok:
            record.update(status="filtered", reason=f"ligand_meeko:{ligand_log}")
            return record
        receptor_ok, receptor_log = _run(
            [
                str(receptor_cli),
                "--read_pdb",
                str(clean_receptor),
                "-p",
                str(receptor_pdbqt),
                "-a",
                "--charge_model",
                "zero",
            ]
        )
        if not receptor_ok:
            record.update(status="filtered", reason=f"receptor_meeko:{receptor_log}")
            return record

        retained_ion_atoms = _append_supported_ions(
            protein_path, receptor_pdbqt
        )

        ligand_atoms, torsions, ligand_types = _pdbqt_summary(ligand_pdbqt)
        receptor_atoms, _, receptor_types = _pdbqt_summary(receptor_pdbqt)
        unsupported_types = sorted(set(ligand_types + receptor_types))
        record.update(
            ligand_pdbqt_atoms=ligand_atoms,
            receptor_pdbqt_atoms=receptor_atoms,
            pdbqt_torsions=torsions,
            retained_receptor_ion_atoms=retained_ion_atoms,
        )
        if unsupported_types:
            record.update(
                status="filtered",
                reason="unsupported_pdbqt_types:" + ",".join(unsupported_types),
            )
            return record
        if torsions == 0:
            record.update(status="filtered", reason="no_rotatable_torsions")
            return record

        system_output = output_dir / system_id
        system_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(receptor_pdbqt, system_output / "receptor.pdbqt")
        shutil.copy2(ligand_pdbqt, system_output / "ligand.pdbqt")

    record.update(status="prepared", primary_benchmark=True)
    return record


def _manifest_summary(records: list[dict]) -> dict:
    selected = [record for record in records if record["journal_subset"]]
    return {
        "source": len(records),
        "journal_subset": len(selected),
        "prepared": sum(record["status"] == "prepared" for record in records),
        "primary_benchmark": sum(record["primary_benchmark"] for record in records),
        "filtered": sum(record["status"] == "filtered" for record in records),
        "excluded_not_in_journal_subset": sum(
            record["status"] == "excluded" for record in records
        ),
    }


def _write_manifest(path: Path, records: list[dict], subset_path: Path) -> None:
    summary = {
        "dataset": "PoseBusters Benchmark Set (journal 308-case subset)",
        "source_url": (
            "https://zenodo.org/records/8278563/files/"
            "posebusters_paper_data.zip"
        ),
        "source_record": "https://zenodo.org/records/8278563",
        "source_archive_md5": SOURCE_ARCHIVE_MD5,
        "source_cases": 428,
        "journal_subset_url": (
            "https://github.com/maabuu/posebusters/files/14516485/"
            "posebusters_pdb_ccd_ids.txt"
        ),
        "journal_subset_sha256": sha256sum(subset_path),
        "license": "CC-BY-4.0",
        "preparation": {
            "meeko": "0.7.1",
            "rdkit": "2026.3.3",
            "charge_model": "zero (charges are unused by the local Vina potential)",
            "pocket_radius_angstrom": POCKET_RADIUS,
            "exclude_macrocycle_ring_size_at_least": 7,
            "removed_pocket_hetero_residue_names": sorted(
                REMOVABLE_POCKET_HETERO_RESIDUES
            ),
            "retained_monoatomic_ion_types": SUPPORTED_MONOATOMIC_ION_TYPES,
            "torsion_range": None,
            "simple_system_filter": False,
            "fixed_receptor_and_ligand_rigid_body": True,
        },
        "counts": _manifest_summary(records),
        "primary_system_ids": [
            record["system_id"] for record in records if record["primary_benchmark"]
        ],
        "systems": records,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest_csv(path.with_suffix(".csv"), records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    binary_dir = Path(sys.executable).parent
    ligand_cli = binary_dir / "mk_prepare_ligand.py"
    receptor_cli = binary_dir / "mk_prepare_receptor.py"
    if not ligand_cli.exists() or not receptor_cli.exists():
        raise RuntimeError(
            "run this script with the isolated preparation environment; "
            "see README.md"
        )
    subset_ids = load_subset_ids(args.subset)
    subset = set(subset_ids)
    source_dirs = sorted(
        path
        for path in args.source.iterdir()
        if path.is_dir() and SYSTEM_ID_PATTERN.fullmatch(path.name)
    )
    observed_ids = {path.name for path in source_dirs}
    missing = sorted(subset - observed_ids)
    if missing:
        raise ValueError(f"journal subset missing from source archive: {missing[:10]}")
    if len(source_dirs) != 428:
        raise ValueError(f"expected 428 source systems, found {len(source_dirs)}")

    records = []
    selected_processed = 0
    selected_total = min(len(subset), args.limit or len(subset))
    for source_dir in source_dirs:
        in_subset = source_dir.name in subset
        if in_subset and args.limit is not None and selected_processed >= args.limit:
            continue
        record = prepare_system(
            source_dir,
            args.output,
            ligand_cli,
            receptor_cli,
            in_subset=in_subset,
        )
        records.append(record)
        if in_subset:
            selected_processed += 1
            print(
                f"[{selected_processed:03d}/{selected_total:03d}] "
                f"{source_dir.name} {record['status']} "
                f"torsions={record['pdbqt_torsions']} reason={record['reason']}",
                flush=True,
            )
    if args.limit is not None:
        records = [record for record in records if record["journal_subset"]]
    _write_manifest(args.manifest, records, args.subset)
    print(json.dumps(json.loads(args.manifest.read_text())["counts"], indent=2))


if __name__ == "__main__":
    main()
