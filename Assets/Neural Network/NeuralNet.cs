using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class NeuralNet : MonoBehaviour
{
     Layer[] layers;
     float learningRate;

     public enum DatasetType { Input, Output };

     // Path file containing the weights of saved Neural Network
     string NNStateFileDirectory = Directory.GetCurrentDirectory() + @"\Assets\Neural Network\2D predictor.txt";

     public NeuralNet(int[] layer, float learningRate)
     {
          layers = new Layer[layer.Length - 1];

          for (int i = 0; i < layers.Length; i++)
          {
               layers[i] = new Layer(layer[i], layer[i + 1]);
          }
          this.learningRate = learningRate;
     }

     ~NeuralNet()
     {
          SaveNN();
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

    // Propagacja wsteczna (nauka sieci wg tego co oczekujemy)
     public void BackProp(float[] expected)
     {
          for (int i = layers.Length - 1; i >= 0; i--)
          {
               if (i == layers.Length - 1) // jeśli obecna warstwa jest ostatnią, to nie ma z czego podać wstecz
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

     // Restoring Neural Network from a .txt file, the neural network MUST BE INITIALIZED WITH TABLES OF APPROPRIATE SIZE
     public void RestoreNN()
     {
          if (File.Exists(NNStateFileDirectory))
          {
               // contents of the .txt file
               string contents = File.ReadAllText(NNStateFileDirectory);
               // string delimiter, when this char appears in string, it gets split
               char separator = ' ';

               // table of weights
               String[] weightsTab = contents.Split(separator);
               // value used to iterate through the weightsTab
               int l = 0;

               // set the weights in the neural network
               for (int i = 0; i < layers.Length; i++)
               {
                    for (int j = 0; j < layers[i].numberOfInputs; j++)
                         for (int k = 0; k < layers[i].numberOfOutputs; k++)
                         {
                              layers[i].weights[k, j] = float.Parse(weightsTab[l++]);
                         }
               }
          }
     }

     // Saving a Neural Network state to a .txt file, not used currently
     public void SaveNN()
     {
          // If file exist delete it, we'll create a new one
          if (File.Exists(NNStateFileDirectory))
          {
               File.Delete(NNStateFileDirectory);
          }

          // String to which we save all the nessessary info about NN
          string contents = "";

          // Iterate through layers, get all needed info (weights)
          for (int i = 0; i < layers.Length; i++)
          {
               for (int j = 0; j < layers[i].numberOfInputs; j++)
                    for (int k = 0; k < layers[i].numberOfOutputs; k++)
                    {
                         contents += layers[i].weights[k, j].ToString();
                         contents += " ";
                    }
          }

          // Save all the info to a file
          File.WriteAllText(NNStateFileDirectory, contents);
     }

     public class Layer
     {
          public int numberOfInputs; // liczba neuronów w poprzedniej warstwie
          public int numberOfOutputs; // liczba neuronów w obecnej warstwie
   
          public float[] outputs;
          public float[] inputs;
          public float[,] weights;
          public float[,] weightsDelta;
          public float[] gamma;
          public float[] error;
          public static System.Random random = new System.Random(); // dałem System. bo Unity ma swoją wersję Randoma i była kolizja namespaców
          
          public Layer(int numberOfInputs, int numberOfOutputs)
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

          public void backPropHidden(float[] gammaForward, float[,] weightsForward)
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

               //Caluclating delta weights
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
