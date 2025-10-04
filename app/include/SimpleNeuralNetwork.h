#ifndef SIMPLENEURALNETWORK_H
#define SIMPLENEURALNETWORK_H


// Enumeration for layer types
enum LayerType {
    FullyConnected = 0,
    Convolutional,
    Pooling,
    Dropout,
    BatchNormalization,
    Flatten,
    ReLU,
	LeakyReLU, 
	PreLU,
    Softmax,
    Sigmoid,
    Tanh    
};

enum UType {
	U_INT8 = 0,
	U_INT16,
	U_INT32,
	U_INT64,
	INT8,
	INT16,
	INT32,
	INT64,
	FLOAT16,
	FLOAT32,
	FLOAT64
};

typedef struct{
	size_t		inputSize;
	size_t		outputSize;
	LayerType	layerType;
	UType		dataType; // Data type of the layer (e.g., float, int)
	void*		layerParams; // Pointer to layer-specific parameters
	void*		weights; // Pointer to weights (if applicable)
	void*		biases;  // Pointer to biases (if applicable)

} layer;


typedef struct{
	size_t	inputSize;
	size_t	hiddenSize;
	size_t	outputSize;

	size_t	numLayers;
	layer* layers;



} SimpleNeuralNetwork_t;;

SimpleNeuralNetwork_t createSimpleNeuralNetwork(int inputSize, int hiddenSize, int outputSize);

#endif

