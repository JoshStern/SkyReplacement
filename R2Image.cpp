// Source file for image class



// Include files 

#include "R2/R2.h"
#include "R2Pixel.h"
#include "R2Image.h"
#include "svd.h"
#include <thread>
#include <iostream>

using namespace std;



////////////////////////////////////////////////////////////////////////
// Constructors/Destructors
////////////////////////////////////////////////////////////////////////


R2Image::
R2Image(void)
  : pixels(NULL),
    npixels(0),
    width(0), 
    height(0)
{
}



R2Image::
R2Image(const char *filename)
  : pixels(NULL),
    npixels(0),
    width(0), 
    height(0)
{
  // Read image
  Read(filename);
}



R2Image::
R2Image(int width, int height)
  : pixels(NULL),
    npixels(width * height),
    width(width), 
    height(height)
{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);
}



R2Image::
R2Image(int width, int height, const R2Pixel *p)
  : pixels(NULL),
    npixels(width * height),
    width(width), 
    height(height)
{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels 
  for (int i = 0; i < npixels; i++) 
    pixels[i] = p[i];
}



R2Image::
R2Image(const R2Image& image)
  : pixels(NULL),
    npixels(image.npixels),
    width(image.width), 
    height(image.height)
    
{
  // Allocate pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels 
  for (int i = 0; i < npixels; i++) 
    pixels[i] = image.pixels[i];
}



R2Image::
~R2Image(void)
{
  // Free image pixels
  if (pixels) delete [] pixels;
}



R2Image& R2Image::
operator=(const R2Image& image)
{
  // Delete previous pixels
  if (pixels) { delete [] pixels; pixels = NULL; }

  // Reset width and height
  npixels = image.npixels;
  width = image.width;
  height = image.height;

  // Allocate new pixels
  pixels = new R2Pixel [ npixels ];
  assert(pixels);

  // Copy pixels 
  for (int i = 0; i < npixels; i++) 
    pixels[i] = image.pixels[i];

  // Return image
  return *this;
}

void R2Image::
svdTest(void)
{
	// fit a 2D conic to five points
	R2Point p1(1.2,3.5);
	R2Point p2(2.1,2.2);
	R2Point p3(0.2,1.6);
	R2Point p4(0.0,0.5);
	R2Point p5(-0.2,4.2);

	// build the 5x6 matrix of equations
	double** linEquations = dmatrix(1,5,1,6);

	linEquations[1][1] = p1[0]*p1[0];
	linEquations[1][2] = p1[0]*p1[1];
	linEquations[1][3] = p1[1]*p1[1];
	linEquations[1][4] = p1[0];
	linEquations[1][5] = p1[1];
	linEquations[1][6] = 1.0;

	linEquations[2][1] = p2[0]*p2[0];
	linEquations[2][2] = p2[0]*p2[1];
	linEquations[2][3] = p2[1]*p2[1];
	linEquations[2][4] = p2[0];
	linEquations[2][5] = p2[1];
	linEquations[2][6] = 1.0;

	linEquations[3][1] = p3[0]*p3[0];
	linEquations[3][2] = p3[0]*p3[1];
	linEquations[3][3] = p3[1]*p3[1];
	linEquations[3][4] = p3[0];
	linEquations[3][5] = p3[1];
	linEquations[3][6] = 1.0;
	
	linEquations[4][1] = p4[0]*p4[0];
	linEquations[4][2] = p4[0]*p4[1];
	linEquations[4][3] = p4[1]*p4[1];
	linEquations[4][4] = p4[0];
	linEquations[4][5] = p4[1];
	linEquations[4][6] = 1.0;

	linEquations[5][1] = p5[0]*p5[0];
	linEquations[5][2] = p5[0]*p5[1];
	linEquations[5][3] = p5[1]*p5[1];
	linEquations[5][4] = p5[0];
	linEquations[5][5] = p5[1];
	linEquations[5][6] = 1.0;

	printf("\n Fitting a conic to five points:\n");
	printf("Point #1: %f,%f\n",p1[0],p1[1]);
	printf("Point #2: %f,%f\n",p2[0],p2[1]);
	printf("Point #3: %f,%f\n",p3[0],p3[1]);
	printf("Point #4: %f,%f\n",p4[0],p4[1]);
	printf("Point #5: %f,%f\n",p5[0],p5[1]);

	// compute the SVD
	double singularValues[7]; // 1..6
	double** nullspaceMatrix = dmatrix(1,6,1,6);
	svdcmp(linEquations, 5, 6, singularValues, nullspaceMatrix);

	// get the result
	printf("\n Singular values: %f, %f, %f, %f, %f, %f\n",singularValues[1],singularValues[2],singularValues[3],singularValues[4],singularValues[5],singularValues[6]);

	// find the smallest singular value:
	int smallestIndex = 1;
	for(int i=2;i<7;i++) if(singularValues[i]<singularValues[smallestIndex]) smallestIndex=i;

	// solution is the nullspace of the matrix, which is the column in V corresponding to the smallest singular value (which should be 0)
	printf("Conic coefficients: %f, %f, %f, %f, %f, %f\n",nullspaceMatrix[1][smallestIndex],nullspaceMatrix[2][smallestIndex],nullspaceMatrix[3][smallestIndex],nullspaceMatrix[4][smallestIndex],nullspaceMatrix[5][smallestIndex],nullspaceMatrix[6][smallestIndex]);

	// make sure the solution is correct:
	printf("Equation #1 result: %f\n",	p1[0]*p1[0]*nullspaceMatrix[1][smallestIndex] + 
										p1[0]*p1[1]*nullspaceMatrix[2][smallestIndex] + 
										p1[1]*p1[1]*nullspaceMatrix[3][smallestIndex] + 
										p1[0]*nullspaceMatrix[4][smallestIndex] + 
										p1[1]*nullspaceMatrix[5][smallestIndex] + 
										nullspaceMatrix[6][smallestIndex]);

	printf("Equation #2 result: %f\n",	p2[0]*p2[0]*nullspaceMatrix[1][smallestIndex] + 
										p2[0]*p2[1]*nullspaceMatrix[2][smallestIndex] + 
										p2[1]*p2[1]*nullspaceMatrix[3][smallestIndex] + 
										p2[0]*nullspaceMatrix[4][smallestIndex] + 
										p2[1]*nullspaceMatrix[5][smallestIndex] + 
										nullspaceMatrix[6][smallestIndex]);

	printf("Equation #3 result: %f\n",	p3[0]*p3[0]*nullspaceMatrix[1][smallestIndex] + 
										p3[0]*p3[1]*nullspaceMatrix[2][smallestIndex] + 
										p3[1]*p3[1]*nullspaceMatrix[3][smallestIndex] + 
										p3[0]*nullspaceMatrix[4][smallestIndex] + 
										p3[1]*nullspaceMatrix[5][smallestIndex] + 
										nullspaceMatrix[6][smallestIndex]);

	printf("Equation #4 result: %f\n",	p4[0]*p4[0]*nullspaceMatrix[1][smallestIndex] + 
										p4[0]*p4[1]*nullspaceMatrix[2][smallestIndex] + 
										p4[1]*p4[1]*nullspaceMatrix[3][smallestIndex] + 
										p4[0]*nullspaceMatrix[4][smallestIndex] + 
										p4[1]*nullspaceMatrix[5][smallestIndex] + 
										nullspaceMatrix[6][smallestIndex]);

	printf("Equation #5 result: %f\n",	p5[0]*p5[0]*nullspaceMatrix[1][smallestIndex] + 
										p5[0]*p5[1]*nullspaceMatrix[2][smallestIndex] + 
										p5[1]*p5[1]*nullspaceMatrix[3][smallestIndex] + 
										p5[0]*nullspaceMatrix[4][smallestIndex] + 
										p5[1]*nullspaceMatrix[5][smallestIndex] + 
										nullspaceMatrix[6][smallestIndex]);

	R2Point test_point(0.34,-2.8);

	printf("A point off the conic: %f\n",	test_point[0]*test_point[0]*nullspaceMatrix[1][smallestIndex] + 
											test_point[0]*test_point[1]*nullspaceMatrix[2][smallestIndex] + 
											test_point[1]*test_point[1]*nullspaceMatrix[3][smallestIndex] + 
											test_point[0]*nullspaceMatrix[4][smallestIndex] + 
											test_point[1]*nullspaceMatrix[5][smallestIndex] + 
											nullspaceMatrix[6][smallestIndex]);

	return;	
}


double* R2Image::constructHomographyMat(double** A, double** AMatch, double** M, double** nullspaceMatrix){
  double sv[10];
  int i, j,k;

  double* H = new double[9];

  //Compute odd rows:
  for(j=1; j<=8; j+=2) {
    for(i=1; i<=3; i++)
      M[j][i] = 0.0;
    for(i=4,k=1; i<=6; i++,k++)
      M[j][i] = (-AMatch[(j+1)/2][3])*A[(j+1)/2][k];
    for(i=7,k=1; i<=9; i++,k++)
      M[j][i] = (AMatch[(j+1)/2][2])*A[(j+1)/2][k];
  }
  //Compute Even rows
  for(j=2; j<=8; j+=2) {
    for(i=1,k=1; i<=3; i++,k++)
      M[j][i] = (AMatch[j/2][3])*A[j/2][k];
    for(i=4; i<=6; i++)
      M[j][i] = 0;
    for(i=7,k=1; i<=9; i++,k++)
      M[j][i] = (-AMatch[j/2][1])*A[j/2][k];      
  }

  // compute the SVD
  svdcmp(M, 8, 9, sv, nullspaceMatrix);

  //Find min 
  int minsvIndex = 1;
  for(i=2; i<10; i++) {
    if(sv[i] < sv[minsvIndex])
      minsvIndex = i;
  }

  for(i=0; i<9; i++) {
    H[i] = nullspaceMatrix[i+1][minsvIndex];
  }
  return H;

}



////////////////////////////////////////////////////////////////////////
// Image processing functions
// YOU IMPLEMENT THE FUNCTIONS IN THIS SECTION
////////////////////////////////////////////////////////////////////////

// Per-pixel Operations ////////////////////////////////////////////////

void R2Image::
Brighten(double factor)
{
  // Brighten the image by multiplying each pixel component by the factor.
  // This is implemented for you as an example of how to access and set pixels
  for (int i = 0; i < width; i++) {
    for (int j = 0;  j < height; j++) {
      Pixel(i,j) *= factor;
      Pixel(i,j).Clamp();
    }
  }
}

void R2Image::
BinaryThreshold() {

  //First blur the image:
  this->Blur(5.0); 

  R2Pixel thresh(0,0,0,1);
  //Do a general search of image to find a decent max pixel
  /*for (int i = 0; i < width; i+=10) {
    for (int j = 0;  j < height; j+=10) {
      if(Pixel(i,j) > thresh)
        thresh = Pixel(i,j);
    }
  }*/

  float blue_thresh = 0.0f; 
  // only works for blue skys... but...

  // search for blue thresh
  for (int i = 0; i < width; i+=10) {
    for (int j = 0;  j < height; j+=10) {
      if(Pixel(i,j).Blue() > blue_thresh)
        blue_thresh = Pixel(i,j).Blue();
    }
  }

  blue_thresh *= 0.8; 
  //thresh *= 0.2;
  //printf("THRESHOLD VALUE FOUND: (%f, %f, %f)\n", thresh.Red(), thresh.Green(), thresh.Blue());

  printf("THRESHOLD BLUE: %f\n", blue_thresh); 

  float th;
  float bl;
  R2Pixel white(1, 1, 1, 1); 
  R2Pixel blk(0, 0, 0, 1); 

  for (int i = 0; i < width; i++) {
    for (int j = 0;  j < height; j++) {
      bl = (float) Pixel(i,j).Blue(); 
      //if(Pixel(i,j) > thresh)
      if(bl > blue_thresh)
        SetPixel(i,j, white);
      else
        SetPixel(i,j, blk);
    }
  }
}

void R2Image::
SkyReplace(vector<R2Image*>* imageList) {


  vector<R2Image*> images = *imageList;

  R2Image* binaryImage = new R2Image(*images[0]);

  double* H = new double[9];

  int* prevPointList = new int[50];
  int* currentPointList = new int[50];
  double x, y, z;

  images.at(0)->Feature(2, prevPointList, 50);
  H = images.at(0)->TrackPoints(prevPointList, 50, images.at(1), currentPointList);

  binaryImage->BinaryThreshold();
  for(int j=0; j<images.at(0)->width; j++) {
    for(int k=0;k<images.at(0)->height;k++) {
      if(binaryImage->Pixel(j,k) == R2Pixel(1,1,1,1)) {
        images.at(0)->SetPixel(j,k, Pixel(j,k));
      }
    }
  }


  // for(int j=0; j<images.at(0)->width; j++) {
  //   for(int k=0;k<images.at(0)->height;k++) {
  //     if(binaryImage->Pixel(j,k) == R2Pixel(1,1,1,1)) {
  //       x = j*H[0] + k*H[1] + 1*H[2];
  //       y = j*H[3] + k*H[4] + 1*H[5];
  //       z = j*H[6] + k*H[7] + 1*H[8];

  //       x /= z; y /=z;

  //       images.at(1)->SetPixel(j,k, Pixel((int)x,(int)y));
  //     }
  //   }
  // }

  delete [] H;
  delete binaryImage;
  delete [] prevPointList;
  delete [] currentPointList;
}

double* R2Image::
TrackPoints(int* points, int size, R2Image* otherImage, int* outPoints) {

  const int KERNEL_SIZE = 4;
  const int KERNEL_MULTIPLIER = 4;
  int cX, cY, minPoint;
  int c,p,k,l, i;
  double ssd, minSSD;

  R2Pixel diff;
  //Go through, checking all 150 points and the surrounding area 
  for(i = 0; i < size; i++) {

    //Find current x and y
    cX = points[i] % width;
    cY = points[i] / width;

    minSSD = 5000.0;
    minPoint = 0;

    // Check out a 20% section of the image, starting from center and moving out
    for(c = -(height/10); c < (height/10); c++) {
      for(p = -(width/10); p < (width/10); p++) {
        // Calculate ssd over kernel
        ssd = 0;
        for(k = -KERNEL_SIZE*KERNEL_MULTIPLIER; k < KERNEL_SIZE*KERNEL_MULTIPLIER; k += KERNEL_MULTIPLIER) {
          for(l = -KERNEL_SIZE*KERNEL_MULTIPLIER; l < KERNEL_SIZE*KERNEL_MULTIPLIER; l += KERNEL_MULTIPLIER) {
            if(cY + k < 0 || cY + k > height-1 || cY + c + k < 0 || cY + c + k > height-1
              || cX + l < 0 || cX + l > width-1 || cX + p + l < 0 || cX + p + l > width-1) {
              ssd += 1.0;
            }
            else {
              diff = Pixel(cX + l, cY + k) - otherImage->Pixel(cX + p + l, cY + c + k);
              ssd +=  diff.Red()*diff.Red() + diff.Green()*diff.Green() + diff.Blue()*diff.Blue();
            }
          }
        }
        if(ssd < minSSD) {
          minSSD = ssd;
          minPoint = (cX + p) + (cY + c)*width;
        }
      }
    }
    outPoints[i] = minPoint;
  }

  return HomogRANSAC(points, outPoints, 50);


}

double* R2Image::HomogRANSAC(int* selectedPoints, int* foundPoints, int NSELECTED) {
  const int TRIAL_NUMBER = 200;
  //Allocate necessary memory:
  double** A = dmatrix(1,4,1,3);
  double** AMatch = dmatrix(1,4,1,3);
  double* H = new double[9];
  double** M = dmatrix(1,8,1,9);
  double** nullspaceMatrix = dmatrix(1,9,1,9);
  double* HBest = new double[9];

  int j, i, k, c = 0, maxC = -1;
  int p[4];
  double pCalcX=-1, pCalcY=-1, pCalcZ=-1;
  double pX, pY, diffX, diffY;

  // Loop 50 times
  for(i = 0; i < TRIAL_NUMBER; i++) {
    //Select 4 random valid index
    for(j = 0; j < 4; j++)
      p[j] = rand() % NSELECTED;

    //Fill our A and AMatch matrices
    for(j = 1; j <= 4; j++){
      A[j][1] = (double)(selectedPoints[p[j-1]] % width);
      A[j][2] = (double)(selectedPoints[p[j-1]] / width);
      A[j][3] = 1.0;
      AMatch[j][1] = (double)(foundPoints[p[j-1]] % width);
      AMatch[j][2] = (double)(foundPoints[p[j-1]] / width);
      AMatch[j][3] = 1.0;
    }
    //Find our H matrix
    H = constructHomographyMat(A, AMatch, M, nullspaceMatrix);

    c = 0;
    //Check it against the others
    for(j = 0; j < NSELECTED; j++) {
      pX = (double)(selectedPoints[j] % width);
      pY = (double)(selectedPoints[j] / width);
      pCalcZ = H[6]*pX + H[7]*pY + H[8]*1.0;
      pCalcX = (H[0]*pX + H[1]*pY + H[2]*1.0) / pCalcZ;
      pCalcY = (H[3]*pX + H[4]*pY + H[5]*1.0) / pCalcZ;
      //Calculate distance between calculated points and true points
      diffX = pCalcX - (double)(foundPoints[j] % width);
      diffY = pCalcY - (double)(foundPoints[j] / width);


      //Check how close this distance is to the original, if close enough, count it as a supporter
      if(diffX*diffX + diffY*diffY < 9.0)
        c++;
    }
    // Find biggest inlier
    if(c > maxC){
      maxC = c;
      for(j=0; j<9; j++) {
        HBest[j] = H[j];
      }
    }
  }

  //Run back through, eliminating points that are too far off
  for(j = 0; j < NSELECTED; j++) {
    pX = (double)(selectedPoints[j] % width);
    pY = (double)(selectedPoints[j] / width);
    pCalcZ = HBest[6]*pX + HBest[7]*pY + HBest[8]*1.0;
    pCalcX = (HBest[0]*pX + HBest[1]*pY + HBest[2]*1.0) / pCalcZ;
    pCalcY = (HBest[3]*pX + HBest[4]*pY + HBest[4]*1.0) / pCalcZ;
    diffX = pCalcX - (double)(foundPoints[j] % width);
    diffY = pCalcY - (double)(foundPoints[j] / width);
    // Set them to -1 if they are too far off
    if(diffX*diffX + diffY*diffY > 9.0) {
      selectedPoints[j] = -1;
      foundPoints[j] = -1;
    }
  }

  return HBest;

}

void R2Image::
SobelX(void)
{
	int i,j;

  double kern[3][3] = {{1.0,0.0,-1.0},
                       {2.0,0.0,-2.0},
                       {1.0,0.0,-1.0}};

  R2Image* finalImage = new R2Image(width, height);

  // Set borders of final image
  for(i = 0; i < width; i++)
    finalImage->SetPixel(i,0, Pixel(i,0));
  for(i = 1; i < height; i++)
    finalImage->SetPixel(0,i, Pixel(0,i));

  // Loop through every pixel except the border ones applying convolution operation
  for(int i = 1; i < height-1; i++){
    for(int j = 1; j < width-1; j++){
      finalImage->SetPixel(j, i, 
                          Pixel(j-1, i-1) * kern[0][0] +
                          Pixel(j, i-1) * kern[0][1] +
                          Pixel(j+1, i-1) * kern[0][2] + 
                          Pixel(j-1, i) * kern[1][0] +
                          Pixel(j, i) * kern[1][1] +
                          Pixel(j+1, i) * kern[1][2] +
                          Pixel(j-1, i+1) * kern[2][0] +
                          Pixel(j, i+1) * kern[2][1] +
                          Pixel(j+1, i+1) * kern[2][2]
        );               
    }
  }
  
  for(i = 1; i < height-1; i++){
    for(j = 1; j < width-1; j++){
      SetPixel(j,i, finalImage->Pixel(j,i));
    }
  }

  delete finalImage;
}

void R2Image::
SobelY(void)
{
  int i,j;

  double kern[3][3] = {{1.0,2.0,1.0},
                       {0.0,0.0,0.0},
                       {-1.0,-2.0,-1.0}};

  R2Image* finalImage = new R2Image(width, height);

  // Set borders of final image
  for(i = 0; i < width; i++)
    finalImage->SetPixel(i,0, Pixel(i,0));
  for(i = 1; i < height; i++)
    finalImage->SetPixel(0,i, Pixel(0,i));

  // Loop through every pixel except the border ones applying convolution operation
  for(int i = 1; i < height-1; i++){
    for(int j = 1; j < width-1; j++){
      finalImage->SetPixel(j, i, 
                          Pixel(j-1, i-1) * kern[0][0] +
                          Pixel(j, i-1) * kern[0][1] +
                          Pixel(j+1, i-1) * kern[0][2] + 
                          Pixel(j-1, i) * kern[1][0] +
                          Pixel(j, i) * kern[1][1] +
                          Pixel(j+1, i) * kern[1][2] +
                          Pixel(j-1, i+1) * kern[2][0] +
                          Pixel(j, i+1) * kern[2][1] +
                          Pixel(j+1, i+1) * kern[2][2]
        );             
    }
  }
  
  for(i = 1; i < height-1; i++){
    for(j = 1; j < width-1; j++){
      SetPixel(j,i, finalImage->Pixel(j,i));
    }
  }
  delete finalImage;
}

void R2Image::
LoG(void)
{
  // Apply the LoG oprator to the image
  
  // FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
  fprintf(stderr, "LoG() not implemented\n");
}



void R2Image::
ChangeSaturation(double factor)
{
  // Changes the saturation of an image
  // Find a formula that changes the saturation without affecting the image brightness

  // FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
  fprintf(stderr, "ChangeSaturation(%g) not implemented\n", factor);
}

void R2Image::
BlurXThread(R2Image* input, R2Image* output, double* kern, int size, int startRow, int endRow) {
  int i, j, k;
  R2Pixel tempPix(0,0,0,1);

  //First pass (X axis)
  for(i = startRow; i < endRow; i++){
    for(j = 0; j < input->width; j++){
      tempPix.Reset(0.0,0.0,0.0,1.0);
      for(k = -size + 1; k < size; k++) { 
        if(j+k < 0){ //If its less than zero, assume it's the zero
          tempPix += input->Pixel(0, i) * kern[abs(k)];
        }
        else if(j+k >= input->width){ // if it's greater than the width, just assume it's the edge pixel
          tempPix += input->Pixel((input->width)-1, i) * kern[abs(k)];
        }
        else{
          tempPix += input->Pixel(j+k, i) * kern[abs(k)];
        }
      }    
      output->SetPixel(j, i, tempPix);        
    }
  }
}

void R2Image::
BlurYThread(R2Image* input, R2Image* output, double* kern, int size, int startCol, int endCol) {
  int i, j, k;
  R2Pixel tempPix(0,0,0,1);

  // Second pass (Y axis)
  for(i = 0; i < input->height; i++){
    for(j = startCol; j < endCol; j++){
      tempPix.Reset(0.0,0.0,0.0,1.0);
      for(k = -size + 1; k < size; k++) {
        if(i+k < 0){
          tempPix += input->Pixel(j, 0) * kern[abs(k)];
        }
        else if(i+k >= input->height){
          tempPix += input->Pixel(j, input->height-1) * kern[abs(k)];
        }
        else {
          tempPix += input->Pixel(j, i+k) * kern[abs(k)];
        }
      }    
      output->SetPixel(j, i, tempPix);    
    }
  }
}

// Linear filtering ////////////////////////////////////////////////
void R2Image::
Blur(double sigma)
{


    //Counter variables
  int i, j, k;
  // Temporary image
  R2Image* tempImage = new R2Image(width, height);
  // Temporary pixel
  R2Pixel tempPix(0.0,0.0,0.0,1.0);
  //Size of kernel (one half)
  int size = (int)(3 * sigma + 1);
  double* kern = new double[size];
  double sqrt2Pi = 2.5066283;

  // Calculate Kernel
  for(i = 0; i < size; i++){
    kern[i] = (1.0 / (sigma*sqrt2Pi))*exp(-(i*i) / (2.0 * sigma * sigma));
  }

  //Spawn X pass threads:
  std::thread X1(R2Image::BlurXThread, this, tempImage, kern, size, 0, height/4);
  std::thread X2(R2Image::BlurXThread, this, tempImage, kern, size, (height/4),height/2);
  std::thread X3(R2Image::BlurXThread, this, tempImage, kern, size, (height/2),(3*height)/4);
  std::thread X4(R2Image::BlurXThread, this, tempImage, kern, size, ((3*height)/4),height);

  X1.join();
  X2.join();
  X3.join();
  X4.join();

  //Run over Y 
  std::thread Y1(R2Image::BlurYThread, tempImage, this, kern, size, 0, width/4);
  std::thread Y2(R2Image::BlurYThread, tempImage, this, kern, size, (width/4),width/2);
  std::thread Y3(R2Image::BlurYThread, tempImage, this, kern, size, (width/2),(3*width)/4);
  std::thread Y4(R2Image::BlurYThread, tempImage, this, kern, size, ((3*width)/4),width);

  Y1.join();
  Y2.join();
  Y3.join();
  Y4.join();


  delete [] kern;
  delete tempImage;
}


int R2Image::
Partition(int* pm, int lo, int hi) 
{
  int i = lo-1;
  int j = hi+1;
  int temp = 0;

  R2Pixel& p = Pixel(pm[lo] % width, pm[lo] / width);

  while(1) {
    do
    {
      i++;
    }while(Pixel((pm[i]) % width, (pm[i]) / width) > p);
    do
    {
      j--;
    }while(Pixel((pm[j]) % width, (pm[j]) / width) < p);
    if(i >= j)
      return j;

    temp = pm[i];
    pm[i] = pm[j];
    pm[j] = temp;

  }
}

void R2Image::
QuickSort(int* pm, int lo, int hi) 
{
  if(lo < hi) {
    int p = Partition(pm, lo, hi);
    QuickSort(pm, lo, p);
    QuickSort(pm, p+1, hi);
  }

}

void R2Image::
SelectPoints(int* pm, int* out, int n) {
  int c, p, cX, cY, pmCursor = 0, pointCount = 0;
  R2Pixel black(0.0,0.0,0.0,1.0);

  //Select the 150 points
  while(pointCount < n && pmCursor < npixels) {
    //Select next best pixel value out of our array
    cX = pm[pmCursor] % width;
    cY = pm[pmCursor] / width;
    //Make sure pixel is not black
    if(Pixel(cX, cY) > black) {
      // Set value
      out[pointCount] = cX + cY*width;
      // Black out pixels in harris
      for(c = -20; c < 20; c++) {
        for(p = -20; p < 20; p++) {
          if(cX+p >= 0 && cY+c >= 0 && cX+p < width && cY+c < height){
            SetPixel(cX+p, cY+c, black);
          }
        }
      }
      pointCount++;
    }
    pmCursor++;
  }
}

void R2Image::
Feature(double sigma, int* selectedPoints, int NSELECTED)
{
  int i, c, p;

  int* pointMap = new int[npixels];

  R2Pixel* red = new R2Pixel(1.0,0.0,0.0,1.0);
  R2Image HarrisImage(*this);
  //Allocate space for a coordinate map
  
  for(i = 0; i < npixels; i++) {
    pointMap[i] = i;
  }

  HarrisImage.Harris(sigma);

  HarrisImage.QuickSort(pointMap, 1, npixels-2);
  int x = pointMap[0] % width;
  int y = pointMap[0] / width;

  HarrisImage.SelectPoints(pointMap, selectedPoints, NSELECTED);
  
  delete red;
  delete [] pointMap;
  delete [] selectedPoints;
}


void R2Image::
Harris(double sigma)
{
  int i, j;
  R2Pixel point5(0.5,0.5,0.5,1.0);

  R2Image Iy(*this);
  R2Image Ix(*this);
  R2Image Ix2(width, height);
  R2Image Iy2(width, height);
  R2Image Ixy(width, height);


  Iy.SobelY();
  Ix.SobelX();

  //Square all elements in Iy and Ix
  for(i = 0; i < height; i++) {
    for(j = 0; j < width; j++) {
      Iy2.SetPixel(j,i, Iy.Pixel(j, i)*Iy.Pixel(j, i));
      Ix2.SetPixel(j,i, Ix.Pixel(j, i)*Ix.Pixel(j, i));
      Ixy.SetPixel(j,i, Ix.Pixel(j, i)*Iy.Pixel(j, i));
    }
  }

  //Blur our three temp images
  Ix2.Blur(sigma);
  Iy2.Blur(sigma);
  Ixy.Blur(sigma);

  // Run harris operation
  for(i = 0; i < height; i++) {
    for(j = 0; j < width; j++) {
      SetPixel(j, i, Ix2.Pixel(j,i)*Iy2.Pixel(j,i) - Ixy.Pixel(j,i)*Ixy.Pixel(j,i) - 0.04*((Ix2.Pixel(j,i) + Iy2.Pixel(j,i))*(Ix2.Pixel(j,i) + Iy2.Pixel(j,i))));
    }
  }
}


void R2Image::
Sharpen()
{
  // Sharpen an image using a linear filter. Use a kernel of your choosing.

  // FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
  fprintf(stderr, "Sharpen() not implemented\n");
}


void R2Image::
blendOtherImageTranslated(R2Image * otherImage)
{
	// find at least 100 features on this image, and another 100 on the "otherImage". Based on these,
	// compute the matching translation (pixel precision is OK), and blend the translated "otherImage" 
	// into this image with a 50% opacity.
	fprintf(stderr, "fit other image using translation and blend imageB over imageA\n");
	return;
}

void R2Image::
blendOtherImageHomography(R2Image * otherImage)
{
	// find at least 100 features on this image, and another 100 on the "otherImage". Based on these,
	// compute the matching homography, and blend the transformed "otherImage" into this image with a 50% opacity.
	fprintf(stderr, "fit other image using a homography and blend imageB over imageA\n");
	return;
}

////////////////////////////////////////////////////////////////////////
// I/O Functions
////////////////////////////////////////////////////////////////////////

int R2Image::
Read(const char *filename)
{
  // Initialize everything
  if (pixels) { delete [] pixels; pixels = NULL; }
  npixels = width = height = 0;

  // Parse input filename extension
  char *input_extension;
  if (!(input_extension = (char*)strrchr(filename, '.'))) {
    fprintf(stderr, "Input file has no extension (e.g., .jpg).\n");
    return 0;
  }
  
  // Read file of appropriate type
  if (!strncmp(input_extension, ".bmp", 4)) return ReadBMP(filename);
  else if (!strncmp(input_extension, ".ppm", 4)) return ReadPPM(filename);
  else if (!strncmp(input_extension, ".jpg", 4)) return ReadJPEG(filename);
  else if (!strncmp(input_extension, ".jpeg", 5)) return ReadJPEG(filename);
  
  // Should never get here
  fprintf(stderr, "Unrecognized image file extension");
  return 0;
}



int R2Image::
Write(const char *filename) const
{
  // Parse input filename extension
  char *input_extension;
  if (!(input_extension = (char*)strrchr(filename, '.'))) {
    fprintf(stderr, "Input file has no extension (e.g., .jpg).\n");
    return 0;
  }
  
  // Write file of appropriate type
  if (!strncmp(input_extension, ".bmp", 4)) return WriteBMP(filename);
  else if (!strncmp(input_extension, ".ppm", 4)) return WritePPM(filename, 1);
  else if (!strncmp(input_extension, ".jpg", 5)) return WriteJPEG(filename);
  else if (!strncmp(input_extension, ".jpeg", 5)) return WriteJPEG(filename);

  // Should never get here
  fprintf(stderr, "Unrecognized image file extension");
  return 0;
}



////////////////////////////////////////////////////////////////////////
// BMP I/O
////////////////////////////////////////////////////////////////////////

#if (RN_OS == RN_LINUX) && !WIN32

typedef struct tagBITMAPFILEHEADER {
  unsigned short int bfType;
  unsigned int bfSize;
  unsigned short int bfReserved1;
  unsigned short int bfReserved2;
  unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER {
  unsigned int biSize;
  int biWidth;
  int biHeight;
  unsigned short int biPlanes;
  unsigned short int biBitCount;
  unsigned int biCompression;
  unsigned int biSizeImage;
  int biXPelsPerMeter;
  int biYPelsPerMeter;
  unsigned int biClrUsed;
  unsigned int biClrImportant;
} BITMAPINFOHEADER;

typedef struct tagRGBTRIPLE {
  unsigned char rgbtBlue;
  unsigned char rgbtGreen;
  unsigned char rgbtRed;
} RGBTRIPLE;

typedef struct tagRGBQUAD {
  unsigned char rgbBlue;
  unsigned char rgbGreen;
  unsigned char rgbRed;
  unsigned char rgbReserved;
} RGBQUAD;

#endif

#define BI_RGB        0L
#define BI_RLE8       1L
#define BI_RLE4       2L
#define BI_BITFIELDS  3L

#define BMP_BF_TYPE 0x4D42 /* word BM */
#define BMP_BF_OFF_BITS 54 /* 14 for file header + 40 for info header (not sizeof(), but packed size) */
#define BMP_BI_SIZE 40 /* packed size of info header */


static unsigned short int WordReadLE(FILE *fp)
{
  // Read a unsigned short int from a file in little endian format 
  unsigned short int lsb, msb;
  lsb = getc(fp);
  msb = getc(fp);
  return (msb << 8) | lsb;
}



static void WordWriteLE(unsigned short int x, FILE *fp)
{
  // Write a unsigned short int to a file in little endian format
  unsigned char lsb = (unsigned char) (x & 0x00FF); putc(lsb, fp); 
  unsigned char msb = (unsigned char) (x >> 8); putc(msb, fp);
}



static unsigned int DWordReadLE(FILE *fp)
{
  // Read a unsigned int word from a file in little endian format 
  unsigned int b1 = getc(fp);
  unsigned int b2 = getc(fp);
  unsigned int b3 = getc(fp);
  unsigned int b4 = getc(fp);
  return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



static void DWordWriteLE(unsigned int x, FILE *fp)
{
  // Write a unsigned int to a file in little endian format 
  unsigned char b1 = (x & 0x000000FF); putc(b1, fp);
  unsigned char b2 = ((x >> 8) & 0x000000FF); putc(b2, fp);
  unsigned char b3 = ((x >> 16) & 0x000000FF); putc(b3, fp);
  unsigned char b4 = ((x >> 24) & 0x000000FF); putc(b4, fp);
}



static int LongReadLE(FILE *fp)
{
  // Read a int word from a file in little endian format 
  int b1 = getc(fp);
  int b2 = getc(fp);
  int b3 = getc(fp);
  int b4 = getc(fp);
  return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



static void LongWriteLE(int x, FILE *fp)
{
  // Write a int to a file in little endian format 
  char b1 = (x & 0x000000FF); putc(b1, fp);
  char b2 = ((x >> 8) & 0x000000FF); putc(b2, fp);
  char b3 = ((x >> 16) & 0x000000FF); putc(b3, fp);
  char b4 = ((x >> 24) & 0x000000FF); putc(b4, fp);
}



int R2Image::
ReadBMP(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s", filename);
    return 0;
  }

  /* Read file header */
  BITMAPFILEHEADER bmfh;
  bmfh.bfType = WordReadLE(fp);
  bmfh.bfSize = DWordReadLE(fp);
  bmfh.bfReserved1 = WordReadLE(fp);
  bmfh.bfReserved2 = WordReadLE(fp);
  bmfh.bfOffBits = DWordReadLE(fp);
  
  /* Check file header */
  assert(bmfh.bfType == BMP_BF_TYPE);
  /* ignore bmfh.bfSize */
  /* ignore bmfh.bfReserved1 */
  /* ignore bmfh.bfReserved2 */
  assert(bmfh.bfOffBits == BMP_BF_OFF_BITS);
  
  /* Read info header */
  BITMAPINFOHEADER bmih;
  bmih.biSize = DWordReadLE(fp);
  bmih.biWidth = LongReadLE(fp);
  bmih.biHeight = LongReadLE(fp);
  bmih.biPlanes = WordReadLE(fp);
  bmih.biBitCount = WordReadLE(fp);
  bmih.biCompression = DWordReadLE(fp);
  bmih.biSizeImage = DWordReadLE(fp);
  bmih.biXPelsPerMeter = LongReadLE(fp);
  bmih.biYPelsPerMeter = LongReadLE(fp);
  bmih.biClrUsed = DWordReadLE(fp);
  bmih.biClrImportant = DWordReadLE(fp);
  
  // Check info header 
  assert(bmih.biSize == BMP_BI_SIZE);
  assert(bmih.biWidth > 0);
  assert(bmih.biHeight > 0);
  assert(bmih.biPlanes == 1);
  assert(bmih.biBitCount == 24);  /* RGB */
  assert(bmih.biCompression == BI_RGB);   /* RGB */
  int lineLength = bmih.biWidth * 3;  /* RGB */
  if ((lineLength % 4) != 0) lineLength = (lineLength / 4 + 1) * 4;
  assert(bmih.biSizeImage == (unsigned int) lineLength * (unsigned int) bmih.biHeight);

  // Assign width, height, and number of pixels
  width = bmih.biWidth;
  height = bmih.biHeight;
  npixels = width * height;

  // Allocate unsigned char buffer for reading pixels
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = bmih.biSizeImage;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
    fprintf(stderr, "Unable to allocate temporary memory for BMP file");
    fclose(fp);
    return 0;
  }

  // Read buffer 
  fseek(fp, (long) bmfh.bfOffBits, SEEK_SET);
  if (fread(buffer, 1, bmih.biSizeImage, fp) != bmih.biSizeImage) {
    fprintf(stderr, "Error while reading BMP file %s", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Allocate pixels for image
  pixels = new R2Pixel [ width * height ];
  if (!pixels) {
    fprintf(stderr, "Unable to allocate memory for BMP file");
    fclose(fp);
    return 0;
  }

  // Assign pixels
  for (int j = 0; j < height; j++) {
    unsigned char *p = &buffer[j * rowsize];
    for (int i = 0; i < width; i++) {
      double b = (double) *(p++) / 255;
      double g = (double) *(p++) / 255;
      double r = (double) *(p++) / 255;
      R2Pixel pixel(r, g, b, 1);
      SetPixel(i, j, pixel);
    }
  }

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return success
  return 1;
}



int R2Image::
WriteBMP(const char *filename) const
{
  // Open file
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s", filename);
    return 0;
  }

  // Compute number of bytes in row
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;

  // Write file header 
  BITMAPFILEHEADER bmfh;
  bmfh.bfType = BMP_BF_TYPE;
  bmfh.bfSize = BMP_BF_OFF_BITS + rowsize * height;
  bmfh.bfReserved1 = 0;
  bmfh.bfReserved2 = 0;
  bmfh.bfOffBits = BMP_BF_OFF_BITS;
  WordWriteLE(bmfh.bfType, fp);
  DWordWriteLE(bmfh.bfSize, fp);
  WordWriteLE(bmfh.bfReserved1, fp);
  WordWriteLE(bmfh.bfReserved2, fp);
  DWordWriteLE(bmfh.bfOffBits, fp);

  // Write info header 
  BITMAPINFOHEADER bmih;
  bmih.biSize = BMP_BI_SIZE;
  bmih.biWidth = width;
  bmih.biHeight = height;
  bmih.biPlanes = 1;
  bmih.biBitCount = 24;       /* RGB */
  bmih.biCompression = BI_RGB;    /* RGB */
  bmih.biSizeImage = rowsize * (unsigned int) bmih.biHeight;  /* RGB */
  bmih.biXPelsPerMeter = 2925;
  bmih.biYPelsPerMeter = 2925;
  bmih.biClrUsed = 0;
  bmih.biClrImportant = 0;
  DWordWriteLE(bmih.biSize, fp);
  LongWriteLE(bmih.biWidth, fp);
  LongWriteLE(bmih.biHeight, fp);
  WordWriteLE(bmih.biPlanes, fp);
  WordWriteLE(bmih.biBitCount, fp);
  DWordWriteLE(bmih.biCompression, fp);
  DWordWriteLE(bmih.biSizeImage, fp);
  LongWriteLE(bmih.biXPelsPerMeter, fp);
  LongWriteLE(bmih.biYPelsPerMeter, fp);
  DWordWriteLE(bmih.biClrUsed, fp);
  DWordWriteLE(bmih.biClrImportant, fp);

  // Write image, swapping blue and red in each pixel
  int pad = rowsize - width * 3;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      const R2Pixel& pixel = (*this)[i][j];
      double r = 255.0 * pixel.Red();
      double g = 255.0 * pixel.Green();
      double b = 255.0 * pixel.Blue();
      if (r >= 255) r = 255;
      if (g >= 255) g = 255;
      if (b >= 255) b = 255;
      fputc((unsigned char) b, fp);
      fputc((unsigned char) g, fp);
      fputc((unsigned char) r, fp);
    }

    // Pad row
    for (int i = 0; i < pad; i++) fputc(0, fp);
  }
  
  // Close file
  fclose(fp);

  // Return success
  return 1;  
}



////////////////////////////////////////////////////////////////////////
// PPM I/O
////////////////////////////////////////////////////////////////////////

int R2Image::
ReadPPM(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s", filename);
    return 0;
  }

  // Read PPM file magic identifier
  char buffer[128];
  if (!fgets(buffer, 128, fp)) {
    fprintf(stderr, "Unable to read magic id in PPM file");
    fclose(fp);
    return 0;
  }

  // skip comments
  int c = getc(fp);
  while (c == '#') {
    while (c != '\n') c = getc(fp);
    c = getc(fp);
  }
  ungetc(c, fp);

  // Read width and height
  if (fscanf(fp, "%d%d", &width, &height) != 2) {
    fprintf(stderr, "Unable to read width and height in PPM file");
    fclose(fp);
    return 0;
  }
	
  // Read max value
  double max_value;
  if (fscanf(fp, "%lf", &max_value) != 1) {
    fprintf(stderr, "Unable to read max_value in PPM file");
    fclose(fp);
    return 0;
  }
	
  // Allocate image pixels
  pixels = new R2Pixel [ width * height ];
  if (!pixels) {
    fprintf(stderr, "Unable to allocate memory for PPM file");
    fclose(fp);
    return 0;
  }

  // Check if raw or ascii file
  if (!strcmp(buffer, "P6\n")) {
    // Read up to one character of whitespace (\n) after max_value
    int c = getc(fp);
    if (!isspace(c)) putc(c, fp);

    // Read raw image data 
    // First ppm pixel is top-left, so read in opposite scan-line order
    for (int j = height-1; j >= 0; j--) {
      for (int i = 0; i < width; i++) {
        double r = (double) getc(fp) / max_value;
        double g = (double) getc(fp) / max_value;
        double b = (double) getc(fp) / max_value;
        R2Pixel pixel(r, g, b, 1);
        SetPixel(i, j, pixel);
      }
    }
  }
  else {
    // Read asci image data 
    // First ppm pixel is top-left, so read in opposite scan-line order
    for (int j = height-1; j >= 0; j--) {
      for (int i = 0; i < width; i++) {
	// Read pixel values
	int red, green, blue;
	if (fscanf(fp, "%d%d%d", &red, &green, &blue) != 3) {
	  fprintf(stderr, "Unable to read data at (%d,%d) in PPM file", i, j);
	  fclose(fp);
	  return 0;
	}

	// Assign pixel values
	double r = (double) red / max_value;
	double g = (double) green / max_value;
	double b = (double) blue / max_value;
        R2Pixel pixel(r, g, b, 1);
        SetPixel(i, j, pixel);
      }
    }
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R2Image::
WritePPM(const char *filename, int ascii) const
{
  // Check type
  if (ascii) {
    // Open file
    FILE *fp = fopen(filename, "w");
    if (!fp) {
      fprintf(stderr, "Unable to open image file: %s", filename);
      return 0;
    }

    // Print PPM image file 
    // First ppm pixel is top-left, so write in opposite scan-line order
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int j = height-1; j >= 0 ; j--) {
      for (int i = 0; i < width; i++) {
        const R2Pixel& p = (*this)[i][j];
        int r = (int) (255 * p.Red());
        int g = (int) (255 * p.Green());
        int b = (int) (255 * p.Blue());
        fprintf(fp, "%-3d %-3d %-3d  ", r, g, b);
        if (((i+1) % 4) == 0) fprintf(fp, "\n");
      }
      if ((width % 4) != 0) fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    // Close file
    fclose(fp);
  }
  else {
    // Open file
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
      fprintf(stderr, "Unable to open image file: %s", filename);
      return 0;
    }
    
    // Print PPM image file 
    // First ppm pixel is top-left, so write in opposite scan-line order
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int j = height-1; j >= 0 ; j--) {
      for (int i = 0; i < width; i++) {
        const R2Pixel& p = (*this)[i][j];
        int r = (int) (255 * p.Red());
        int g = (int) (255 * p.Green());
        int b = (int) (255 * p.Blue());
        fprintf(fp, "%c%c%c", r, g, b);
      }
    }
    
    // Close file
    fclose(fp);
  }

  // Return success
  return 1;  
}



////////////////////////////////////////////////////////////////////////
// JPEG I/O
////////////////////////////////////////////////////////////////////////


// #define USE_JPEG
#ifdef USE_JPEG
  extern "C" { 
#   define XMD_H // Otherwise, a conflict with INT32
#   undef FAR // Otherwise, a conflict with windows.h
#   include "jpeg/jpeglib.h"
  };
#endif



int R2Image::
ReadJPEG(const char *filename)
{
#ifdef USE_JPEG
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s", filename);
    return 0;
  }

  // Initialize decompression info
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  // Remember image attributes
  width = cinfo.output_width;
  height = cinfo.output_height;
  npixels = width * height;
  int ncomponents = cinfo.output_components;

  // Allocate pixels for image
  pixels = new R2Pixel [ npixels ];
  if (!pixels) {
    fprintf(stderr, "Unable to allocate memory for BMP file");
    fclose(fp);
    return 0;
  }

  // Allocate unsigned char buffer for reading image
  int rowsize = ncomponents * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = rowsize * height;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
    fprintf(stderr, "Unable to allocate temporary memory for JPEG file");
    fclose(fp);
    return 0;
  }

  // Read scan lines 
  // First jpeg pixel is top-left, so read pixels in opposite scan-line order
  while (cinfo.output_scanline < cinfo.output_height) {
    int scanline = cinfo.output_height - cinfo.output_scanline - 1;
    unsigned char *row_pointer = &buffer[scanline * rowsize];
    jpeg_read_scanlines(&cinfo, &row_pointer, 1);
  }

  // Free everything
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  // Close file
  fclose(fp);

  // Assign pixels
  for (int j = 0; j < height; j++) {
    unsigned char *p = &buffer[j * rowsize];
    for (int i = 0; i < width; i++) {
      double r, g, b, a;
      if (ncomponents == 1) {
        r = g = b = (double) *(p++) / 255;
        a = 1;
      }
      else if (ncomponents == 1) {
        r = g = b = (double) *(p++) / 255;
        a = 1;
        p++;
      }
      else if (ncomponents == 3) {
        r = (double) *(p++) / 255;
        g = (double) *(p++) / 255;
        b = (double) *(p++) / 255;
        a = 1;
      }
      else if (ncomponents == 4) {
        r = (double) *(p++) / 255;
        g = (double) *(p++) / 255;
        b = (double) *(p++) / 255;
        a = (double) *(p++) / 255;
      }
      else {
        fprintf(stderr, "Unrecognized number of components in jpeg image: %d\n", ncomponents);
        return 0;
      }
      R2Pixel pixel(r, g, b, a);
      SetPixel(i, j, pixel);
    }
  }

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return success
  return 1;
#else
  fprintf(stderr, "JPEG not supported");
  return 0;
#endif
}


	

int R2Image::
WriteJPEG(const char *filename) const
{
#ifdef USE_JPEG
  // Open file
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open image file: %s", filename);
    return 0;
  }

  // Initialize compression info
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);
  cinfo.image_width = width; 	/* image width and height, in pixels */
  cinfo.image_height = height;
  cinfo.input_components = 3;		/* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */
  cinfo.dct_method = JDCT_ISLOW;
  jpeg_set_defaults(&cinfo);
  cinfo.optimize_coding = TRUE;
  jpeg_set_quality(&cinfo, 95, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
	
  // Allocate unsigned char buffer for reading image
  int rowsize = 3 * width;
  if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
  int nbytes = rowsize * height;
  unsigned char *buffer = new unsigned char [nbytes];
  if (!buffer) {
    fprintf(stderr, "Unable to allocate temporary memory for JPEG file");
    fclose(fp);
    return 0;
  }

  // Fill buffer with pixels
  for (int j = 0; j < height; j++) {
    unsigned char *p = &buffer[j * rowsize];
    for (int i = 0; i < width; i++) {
      const R2Pixel& pixel = (*this)[i][j];
      int r = (int) (255 * pixel.Red());
      int g = (int) (255 * pixel.Green());
      int b = (int) (255 * pixel.Blue());
      if (r > 255) r = 255;
      if (g > 255) g = 255;
      if (b > 255) b = 255;
      *(p++) = r;
      *(p++) = g;
      *(p++) = b;
    }
  }



  // Output scan lines
  // First jpeg pixel is top-left, so write in opposite scan-line order
  while (cinfo.next_scanline < cinfo.image_height) {
    int scanline = cinfo.image_height - cinfo.next_scanline - 1;
    unsigned char *row_pointer = &buffer[scanline * rowsize];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  // Free everything
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  // Close file
  fclose(fp);

  // Free unsigned char buffer for reading pixels
  delete [] buffer;

  // Return number of bytes written
  return 1;
#else
  fprintf(stderr, "JPEG not supported");
  return 0;
#endif
}






