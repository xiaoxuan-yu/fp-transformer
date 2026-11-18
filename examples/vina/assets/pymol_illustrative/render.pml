reinitialize
bg_color white
set ray_opaque_background, off
set orthoscopic, on
set auto_zoom, off
set depth_cue, off
set fog, off
set antialias, 2
set ambient, 0.72
set direct, 0.28
set reflect, 0.06
set specular, 0
set shininess, 0
set ray_shadows, off
set ray_trace_mode, 1
set ray_trace_gain, 0.08
set ray_trace_color, gray60
set_color atom_c, [0.694, 0.655, 0.620]
set_color atom_n, [0.494, 0.576, 0.651]
set_color atom_o, [0.773, 0.561, 0.498]
set_color atom_p, [0.569, 0.639, 0.580]
set_color atom_s, [0.710, 0.616, 0.435]
set_color atom_halogen, [0.569, 0.639, 0.580]
set_color pocket_surface, [0.862745, 0.921569, 0.921569]
set_color torsion_highlight, [0.600, 0.553, 0.647]
set stick_radius, 0.16
set sphere_scale, 0.26
set stick_quality, 14
set sphere_quality, 2
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_best.pdb, fpt_best
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/mc_0.pdb, mc_0
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/mc_1.pdb, mc_1
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/mc_2.pdb, mc_2
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/mc_3.pdb, mc_3
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_0_0.pdb, fpt_0_0
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_0_1.pdb, fpt_0_1
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_0_2.pdb, fpt_0_2
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_1_0.pdb, fpt_1_0
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_1_1.pdb, fpt_1_1
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_1_2.pdb, fpt_1_2
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_2_0.pdb, fpt_2_0
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_2_1.pdb, fpt_2_1
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/fpt_2_2.pdb, fpt_2_2
color atom_c, (all) and elem C
color atom_n, (all) and elem N
color atom_o, (all) and elem O
color atom_p, (all) and elem P
color atom_s, (all) and elem S
color atom_halogen, (all) and elem F+Cl+Br+I
orient all
zoom all, 1.8
hide everything, all
show sticks, fpt_best
show spheres, fpt_best
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_best.png, 520, 520, 300, ray=1
hide everything, fpt_best
show sticks, mc_0
show spheres, mc_0
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/mc_0.png, 520, 520, 300, ray=1
hide everything, mc_0
show sticks, mc_1
show spheres, mc_1
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/mc_1.png, 520, 520, 300, ray=1
hide everything, mc_1
show sticks, mc_2
show spheres, mc_2
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/mc_2.png, 520, 520, 300, ray=1
hide everything, mc_2
show sticks, mc_3
show spheres, mc_3
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/mc_3.png, 520, 520, 300, ray=1
hide everything, mc_3
show sticks, fpt_0_0
show spheres, fpt_0_0
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_0_0.png, 520, 520, 300, ray=1
hide everything, fpt_0_0
show sticks, fpt_0_1
show spheres, fpt_0_1
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_0_1.png, 520, 520, 300, ray=1
hide everything, fpt_0_1
show sticks, fpt_0_2
show spheres, fpt_0_2
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_0_2.png, 520, 520, 300, ray=1
hide everything, fpt_0_2
show sticks, fpt_1_0
show spheres, fpt_1_0
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_1_0.png, 520, 520, 300, ray=1
hide everything, fpt_1_0
show sticks, fpt_1_1
show spheres, fpt_1_1
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_1_1.png, 520, 520, 300, ray=1
hide everything, fpt_1_1
show sticks, fpt_1_2
show spheres, fpt_1_2
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_1_2.png, 520, 520, 300, ray=1
hide everything, fpt_1_2
show sticks, fpt_2_0
show spheres, fpt_2_0
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_2_0.png, 520, 520, 300, ray=1
hide everything, fpt_2_0
show sticks, fpt_2_1
show spheres, fpt_2_1
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_2_1.png, 520, 520, 300, ray=1
hide everything, fpt_2_1
show sticks, fpt_2_2
show spheres, fpt_2_2
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/fpt_2_2.png, 520, 520, 300, ray=1
hide everything, fpt_2_2
delete all
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/data/raw/posebusters_benchmark_set/7ELT_TYM/7ELT_TYM_protein.pdb, protein
load /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/structures/reference.pdb, ligand
remove solvent
select pocket, byres ((polymer.protein and protein) within 9.5 of ligand)
hide everything, all
set ray_trace_mode, 0
set ambient, 0.84
set direct, 0.16
set reflect, 0
set ray_trace_gain, 0
set stick_radius, 0.20, ligand
set sphere_scale, 0.34, ligand
color pocket_surface, pocket
set surface_color, pocket_surface, pocket
set transparency, 0, pocket
set surface_quality, 1
set solvent_radius, 1.4
set surface_mode, 1
color atom_c, (ligand) and elem C
color atom_n, (ligand) and elem N
color atom_o, (ligand) and elem O
color atom_p, (ligand) and elem P
color atom_s, (ligand) and elem S
color atom_halogen, (ligand) and elem F+Cl+Br+I
orient pocket or ligand
zoom pocket or ligand, 0.8
hide everything, all
show surface, pocket
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/system_surface.png, 900, 900, 300, ray=1
hide everything, all
show sticks, ligand
show spheres, ligand
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/system_ligand.png, 900, 900, 300, ray=1
create torsion_marker, ligand and id 19+21
hide everything, all
show sticks, torsion_marker
color torsion_highlight, torsion_marker
set stick_radius, 0.11, torsion_marker
png /home/youmans/fp-transformer/.worktrees/vina-score/examples/case11_vina_posebusters/assets/pymol_illustrative/system_torsion.png, 900, 900, 300, ray=1
quit
