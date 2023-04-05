# miniBatchBasedCudaSLenet
this is the example of cuda simple mini batch based lenet model. However there is still some issues with training accuracy calculation.
# here is the command sequences to run a code 
nvcc -lcuda -lcublas -rdc=true layer.cu main.cu -o CNN -arch=compute_53 -Wno-deprecated-gpu-targets
CNN 1 1
//where the first argument after CNN stands for number of epochs 
//and the second argument after CNN stands for mini batch size, can be up to 32
