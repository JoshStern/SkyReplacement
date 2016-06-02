# SkyReplacement

##Final Project for Computer Vision
Group members: Josh Stern, Savvas Petridis, Eli Fessler

##Objective:
Take in a sequence of JPEG images and replace a high contrast sky with another texture.

##Components
  1. Parallelization of necessary algorithms
  2. Frame to Frame data structure
  3. Sky isolation
  4. Texture application

##Algorithm Flow
1. Threshold each image as to isolate the sky (allowing us to do multiplication for replacement)
2. Track points from one original image to the next
  1. Using these points, calculate homography matrix
3. Use Homography to calculate where to sample from sky texture image (so it moves appropriately)

##Input/Output Details
###One image:
	./imgpro <full_input_image_name> <full_output_image_name> 1 <-flag stuff>
###More than one:
	./imgpro <general_input_image_name> <general_output_image_name> <image_count> <-flag stuff>
This will append "_i.jpg" to all general names, where i is an integer (starting from zero).

##Unexpected Results
If images are flipped (due to extra image metadata), install [Jhead](http://www.sentex.net/~mwandel/jhead/) using ``brew install jhead`` ([Homebrew](http://brew.sh/) required) and run

	jhead -autorot path/to/image.jpg
