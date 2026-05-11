# Limitations

- Monocular reconstruction has inherent scale ambiguity; this project mitigates it with sparse COLMAP alignment but does not guarantee metric scale.
- COLMAP can fail on textureless walls, repeated patterns, motion blur, or videos with too little viewpoint change.
- Learned depth may fail on reflective, transparent, or unusual indoor surfaces.
- Dynamic objects are fused as if static.
- The output coordinate frame is COLMAP world, not a robot base frame or gravity-aligned frame.
- Floor and obstacle cues are conservative metadata in this version, not a full navigation stack.

