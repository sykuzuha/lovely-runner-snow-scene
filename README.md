# lovely-runner-snow-scene
For our project, we have recreated a still scene from the k-drama Lovely Runner with cats instead of humans. In the scene, one cat holds a yellow umbrella over another cat in the snow.

Our project idea is related to computer graphics because it involves rendering a 3-D scene. We will apply concepts we learned like meshes, lighting, camera transformations, shading, and texture mapping. We can use procedural techniques for the snowfall, as well as potentially shading or lighting effects to simulate the snow atmosphere. To complete this project, we will need to learn about particle systems for the snow and maybe some more advanced animation techniques for scene composition.

Languages and middleware:
- C++
- OpenGL
- Blender for modeling/posing the cats and the umbrella

This project currently only works on MacOS with arm64.

To run it, make sure to install the `glfw glm assimp libpng` libraries if you don't already have those dependencies installed (can be done easily with homebrew!) and then the project can be run with:

$ make run

or

$ make
$ ./cat_scene

To change the background music:

The audio file at `assets/music.mp3` will auto-play and loop.
