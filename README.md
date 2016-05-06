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
  5. Actual algorithm
  
##Division of work
TBD!

##Input/Output Details
###One image:
	"./imgpro <full_input_image_name> <full_output_image_name> 1 <-flag stuff>"
###More than one:
	"./imgpro <general_input_image_name> <general_output_image_name> <image_count> <-flag stuff>"  
This will append "_i.jpg" to all general names, where i is an integer (starting from zero).

