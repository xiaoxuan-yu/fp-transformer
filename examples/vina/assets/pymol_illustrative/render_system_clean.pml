reinitialize
bg_color white
set ray_opaque_background, off
set orthoscopic, on
set auto_zoom, off
set depth_cue, off
set fog, off
set antialias, 2
set ray_shadows, off
set specular, 0
set shininess, 0
set ambient, 0.82
set direct, 0.18
set reflect, 0
set ray_trace_gain, 0
set surface_quality, 2
set surface_mode, 0
set solvent_radius, 1.2

set_color pocket_surface_clean, [0.765, 0.855, 0.851]
set_color ligand_outline, [0.180, 0.196, 0.200]
set_color ligand_c_clean, [0.792, 0.408, 0.173]
set_color ligand_n_clean, [0.247, 0.443, 0.635]
set_color ligand_o_clean, [0.737, 0.204, 0.180]
set_color ligand_p_clean, [0.439, 0.557, 0.345]
set_color ligand_s_clean, [0.788, 0.620, 0.180]
set_color ligand_halogen_clean, [0.373, 0.565, 0.365]
set_color torsion_highlight_clean, [0.529, 0.318, 0.604]

load /home/youmans/docs/fpt-paper1/ref/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/data/raw/posebusters_benchmark_set/7ELT_TYM/7ELT_TYM_protein.pdb, protein
load /home/youmans/docs/fpt-paper1/ref/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/reference.pdb, ligand
remove solvent
select pocket, byres ((polymer.protein and protein) within 5.5 of ligand)
color pocket_surface_clean, protein
set surface_color, pocket_surface_clean, protein

color ligand_c_clean, (ligand) and elem C
color ligand_n_clean, (ligand) and elem N
color ligand_o_clean, (ligand) and elem O
color ligand_p_clean, (ligand) and elem P
color ligand_s_clean, (ligand) and elem S
color ligand_halogen_clean, (ligand) and elem F+Cl+Br+I

orient pocket or ligand
zoom pocket or ligand, 1.1

hide everything, all
show surface, pocket
set transparency, 0.40, pocket
png /home/youmans/docs/fpt-paper1/ref/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/system_surface.png, 1200, 1200, 300, ray=1

hide everything, all
show sticks, ligand
show spheres, ligand
color ligand_outline, ligand
set stick_radius, 0.27, ligand
set sphere_scale, 0.46, ligand
png /home/youmans/docs/fpt-paper1/ref/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/system_ligand_outline.png, 1200, 1200, 300, ray=1

color ligand_c_clean, (ligand) and elem C
color ligand_n_clean, (ligand) and elem N
color ligand_o_clean, (ligand) and elem O
color ligand_p_clean, (ligand) and elem P
color ligand_s_clean, (ligand) and elem S
color ligand_halogen_clean, (ligand) and elem F+Cl+Br+I
set stick_radius, 0.20, ligand
set sphere_scale, 0.35, ligand
png /home/youmans/docs/fpt-paper1/ref/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/system_ligand.png, 1200, 1200, 300, ray=1

create torsion_marker, ligand and id 19+21
hide everything, all
show sticks, torsion_marker
color torsion_highlight_clean, torsion_marker
set stick_radius, 0.12, torsion_marker
png /home/youmans/docs/fpt-paper1/ref/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/system_torsion.png, 1200, 1200, 300, ray=1
quit
