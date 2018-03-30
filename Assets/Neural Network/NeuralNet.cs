using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class NeuralNet : MonoBehaviour {

    int[] layer;
    Layer[] layers;
    float learningRate;


    public NeuralNet(int[] layer, float learningRate)
    {
        this.layer = new int[layer.Length];
        for (int i = 0; i < layer.Length; i++)
            this.layer[i] = layer[i];

        layers = new Layer[layer.Length - 1];

        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layer[i], layer[i + 1]);
        }
        this.learningRate = learningRate;
    }

    // Podanie dalej stopniowo całej sieci
    public float[] FeedForward(float[] inputs)
    {
        layers[0].FeedForward(inputs); // pierwsza warstwa jedynie podaje niezmienione wejścia
        for (int i = 1; i < layers.Length; i++)
        {
            layers[i].FeedForward(layers[i - 1].outputs);
        }
        return layers[layers.Length - 1].outputs; // zwrot wyjść ostatniej warstwy
    }

    public void BackProp(float[] expected)
    {
        for (int i = layers.Length - 1; i >= 0; i--)
        {
            if(i == layers.Length - 1) // jeśli obecna warstwa jest ostatnią, to nie ma z czego podać wstecz
                {
                layers[i].backPropOutput(expected);
                }
            else //podajemy w tył wagi i gammy następnych warstw
            {
                layers[i].backPropHidden(layers[i + 1].gamma, layers[i + 1].weights); 
            }
        }

        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].updateWeights(this.learningRate);
        }

    }


	
    public class Layer
    {
        int numberOfInputs; // liczba neuronów w poprzedniej warstwie
        int numberOfOutputs; // liczba neuronów w obecnej warstwie
        

       
       public float[] outputs;
       public float[] inputs;
       public float[ , ] weights;
       public float[ , ] weightsDelta;
       public float[] gamma;
       public float[] error;
       public static System.Random random = new System.Random(); // dałem System. bo Unity ma swoją wersję Randoma i była kolizja namespaców

       

        public Layer( int numberOfInputs, int numberOfOutputs)
        {
            this.numberOfInputs = numberOfInputs;
            this.numberOfOutputs = numberOfOutputs;

            outputs = new float[numberOfOutputs];
            inputs = new float[numberOfInputs];
            weights = new float[numberOfOutputs, numberOfInputs];
            weightsDelta = new float[numberOfOutputs, numberOfInputs];
            gamma = new float[numberOfOutputs];
            error = new float[numberOfOutputs];

            InitializaWeights();
        }

        public void InitializaWeights()
        {
            for (int i = 0; i < numberOfOutputs; i++)
            {
                for (int j = 0; j < numberOfInputs; j++)
                {
                    weights[i, j] = (float)random.NextDouble() - 0.5f;
                }
            }
        }

        // Podanie dalej pojedynczej warstwy
        public float[] FeedForward(float[] inputs)
        {

            this.inputs = inputs;

            for (int i = 0; i < numberOfOutputs; i++)
            {
                outputs[i] = 0;
                for (int j = 0; j < numberOfInputs; j++)
                {
                    outputs[i] += inputs[j] * weights[i, j];
                }

                outputs[i] = (float)Math.Tanh(outputs[i]);
            }

            return outputs;
        }

        // funkcja zwracająca pochodną tanH (sama matma, bez znaczenia dla algorytmu)
        public float TanHDer(float value)
        {
            return 1 - (value * value);
        }

        public void backPropOutput(float[] expected)
        {
            for (int i = 0; i < numberOfOutputs; i++)
                error[i] = outputs[i] - expected[i];

            for (int i = 0; i < numberOfOutputs; i++)
                gamma[i] = error[i] * TanHDer(outputs[i]);



            for (int i = 0; i < numberOfOutputs; i++)
            {
                for (int j = 0; j < numberOfInputs; j++)
                {
                    weightsDelta[i, j] = gamma[i] * inputs[j];
                }
            }

        }

        public void backPropHidden( float[] gammaForward, float[,] weightsForward)
        {
            for (int i = 0; i < numberOfOutputs; i++)
            {
                gamma[i] = 0;

                for (int j = 0; j < gammaForward.Length; j++)
                {
                    gamma[i] += gammaForward[j] * weightsForward[j, i];
                }

                gamma[i] *= TanHDer(outputs[i]);
            }

            //Caluclating detla weights
            for (int i = 0; i < numberOfOutputs; i++)
            {
                for (int j = 0; j < numberOfInputs; j++)
                {
                    weightsDelta[i, j] = gamma[i] * inputs[j];
                }
            }
        }



        public void updateWeights(float learningRate)
        {
            for (int i = 0; i < numberOfOutputs; i++)
            {
                for (int j = 0; j < numberOfInputs; j++)
                {
                    weights[i, j] -= weightsDelta[i, j] * learningRate;
                }
            }
        }

    }
}
