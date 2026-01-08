# Alphabet Data Sources Strategy

## Unicode Font Sources

### High-Priority (Easier to find)
- **Latin**: Any standard serif/sans-serif fonts
- **Greek**: DejaVu Sans, Gentium Plus, New Athena Unicode
- **Armenian**: Sylfaen, Noto Sans Armenian, Arial AMU
- **Georgian**: Noto Sans Georgian, BPG Glaho, Sylfaen

### Medium-Priority (Specialized fonts)
- **Tifinagh**: Noto Sans Tifinagh, IRAM Tifinagh
- **Avestan**: Noto Sans Avestan, Avestan fonts from academic sources

### Hard-to-Find (Academic/historical)
- **Old Permic**: Buran, Permian font packages
- **Phrygian**: Academic reconstruction fonts
- **Coorgi-Cox**: May need custom font creation
- **A-chik Tokbirim**: Indigenous language font projects
- **Zoulai**: Academic linguistic resources
- **Naasioi Otomaung**: PNG linguistic documentation

## Image Sources

### Manuscript & Inscription Collections
- **Georgian**: National Archives of Georgia, medieval manuscripts
- **Armenian**: Matenadaran Institute, medieval Armenian manuscripts
- **Greek**: Papyrus collections, inscription corpora
- **Tifinagh**: Rock art documentation, modern signage

### Academic Resources
- **Unicode Consortium**: Character charts and examples
- **SIL International**: Font development for minority languages
- **Linguistic societies**: Documentation materials

## Data Collection Priority

### Phase 1 (Test Pipeline)
1. Latin (Arial, Times New Roman)
2. Greek (Noto Sans Greek)
3. Armenian (Noto Sans Armenian)

### Phase 2 (Expand)
4. Georgian (Noto Sans Georgian)
5. Tifinagh (Noto Sans Tifinagh)
6. Avestan (Noto Sans Avestan)

### Phase 3 (Specialized)
7. Old Permic
8. Phrygian
9. Székely-Hungarian Rovás
10. Remaining indigenous scripts

## Character Mapping Strategy

### Phonetic Mapping (Primary)
- Greek Α→A, Β→B, Γ→G, Δ→D
- Armenian Ա→A, Բ→B, Գ→G, Դ→D
- Georgian Ⴀ→A, Ⴁ→B, Ⴂ→G (Asomtavruli)

### Visual Similarity (Secondary)
- Round characters: O, Ո, ለ, ჲ
- Linear characters: I, Ի, Ι, Ⴈ
- Complex characters: Q, Ք, ჴ

### Custom Mapping (Advanced)
- Allow user-defined character correspondences
- Cultural/historical mapping preferences
- Script family groupings
