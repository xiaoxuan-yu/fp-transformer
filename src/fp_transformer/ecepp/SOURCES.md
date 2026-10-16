# ECEPP/3 provenance

The functional form and parameter-table definitions follow:

- F. A. Momany, R. F. McGuire, A. W. Burgess, and H. A. Scheraga,
  *J. Phys. Chem.* **79**, 2361–2381 (1975).
- G. Némethy, M. S. Pottle, and H. A. Scheraga,
  *J. Phys. Chem.* **87**, 1883–1887 (1983).
- M. J. Sippl, G. Némethy, and H. A. Scheraga,
  *J. Phys. Chem.* **88**, 6231–6233 (1984).
- G. Némethy et al., *J. Phys. Chem.* **96**, 6472–6484 (1992),
  the ECEPP/3 revision.

The implementation was cross-checked against SMMP 3.0.5:

- project archive: `SMMP-3.0.5.tar.bz2`
- archive SHA-256:
  `70d6b719d2c2ef1d34fd9897ad1c89a37c4d7531d80c904eee39dc8855aca31c`
- `SMMP/lib.sh3` SHA-256:
  `940c406669b24bd06d750c284b5ff4a7b7bde87d57d880c07a4797a5f51a773e`
- `init_energy.f` SHA-256:
  `155b6a6f10d53a20ba4278b9a03e3527359f73c02f5b67783a2a67e6f3624c00`

SMMP's license states GPL-2.0 terms plus additional non-commercial and
non-military restrictions. No SMMP source or `lib.sh3` data is redistributed in
this repository. `ECEPP3Library` only provides an interoperability parser for a
copy obtained separately by the user.
