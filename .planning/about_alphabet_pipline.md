script `alphabet_pipeline.py` contains 9 major sections:

- Font Parsing - TTF → vector paths with composite resolution
- Arc-Length Sampling - Geometric invariance (256 samples default)
- Normalization - Canonical transform (translate → scale)
- Rasterization - 512×512 binary images for analysis
- Skeletonization - Medial axis extraction with validation
- Topology Analysis - Endpoints, junctions, connectivity
- Statistics - Per-glyph and alphabet-level metrics
- Batch Processing - Full pipeline execution
- Main Entry - Command-line interface