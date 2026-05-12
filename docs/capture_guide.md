# Capture Guide

Good video input makes a larger difference than most algorithm tweaks. The pipeline assumes a short indoor monocular video with enough overlap for COLMAP and enough parallax for useful geometry.

## Recommended Capture

- 10-30 seconds.
- Slow lateral motion around the scene.
- Keep the phone steady and avoid sudden rotations.
- Keep the indoor area small, such as a desk corner, room corner, or small room.
- Include textured surfaces, object edges, posters, shelves, or furniture.
- Keep strong overlap between neighboring views.
- Use good lighting and avoid heavy motion blur.

## Avoid

- Pure white walls with very little texture.
- Fast pans or rapid rotations in place.
- Reflective, transparent, or glossy surfaces as the main subject.
- Moving people or moving objects.
- Videos with almost no viewpoint change.
- Very large rooms when using the default lightweight settings.

## Practical Tip

For the strongest demo, move sideways while keeping the main scene in view. A translation-heavy phone video gives COLMAP better parallax than a video captured mostly by rotating the phone from one spot.
