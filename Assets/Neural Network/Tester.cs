using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Tester : MonoBehaviour
{
     // To żeby móc się dostać do tych obiektów
     private Transform playerTransform;
     private Transform ball1;
     private Transform ball2;
     private Transform ball3;

     public static int inputNeuronCount = 20;
     public static int outputNeuronCount = 6;


     float[] lastPositions = new float[inputNeuronCount];
     float[] predictedVectors = new float[outputNeuronCount];
     NeuralNet net;

     // Use this for initialization
     void Start()
     {
          // Statek
          playerTransform = GameObject.Find("Statek").transform;
          // 3 punkty do wizualizacji zmian
          ball1 = GameObject.Find("ball1").transform;
          ball2 = GameObject.Find("ball2").transform;
          ball3 = GameObject.Find("ball3").transform;


          // reset table
          for (int i = 0; i < lastPositions.Length; i++)
               lastPositions[i] = 0.0f;

          net = new NeuralNet(new int[] { inputNeuronCount, 13, outputNeuronCount }, 0.00333f);
          Train(10000, DataType.Random);
     }

     // Update is called once per frame
     void Update()
     {
         
          
               // new position
               float newX = playerTransform.position.x;
               float newY = playerTransform.position.z;

               // nn is taking vectors for input
               float[] lastPosVectors = new float[lastPositions.Length];
               // fill the table with vectors except for 2 last elements
               for(int i = 0; i < lastPosVectors.Length - 2; i++)
               {
                    lastPosVectors[i] = lastPositions[i + 2] - lastPositions[i];
               }
               // fill the last 2 elements
               lastPosVectors[lastPosVectors.Length - 2] = newX - lastPosVectors[lastPosVectors.Length - 2];
               lastPosVectors[lastPosVectors.Length - 1] = newY - lastPosVectors[lastPosVectors.Length - 1];

               // shift table, wipe oldest point and make room for new one
               for (int i = 0; i < lastPositions.Length - 2; i++)
                    lastPositions[i] = lastPositions[i + 2];

               // put the new point into the table
               lastPositions[lastPositions.Length - 2] = newX;
               lastPositions[lastPositions.Length - 1] = newY;
                
               // Get predicted vectors by providing actual vectors
               predictedVectors = Test(lastPosVectors);

               // set ball positions based on our output
               ball1.position = new Vector3(playerTransform.position.x + predictedVectors[0], playerTransform.position.y, playerTransform.position.z + predictedVectors[1]);
               ball2.position = new Vector3(ball1.position.x + predictedVectors[2], ball1.position.y, ball1.position.z + predictedVectors[3]);
               ball3.position = new Vector3(ball2.position.x + predictedVectors[4], ball2.position.y, ball2.position.z + predictedVectors[5]);

     }

     private void OnGUI()
     {
          String msg = "";
          for (int i = 0; i < 6; i+=2)
               msg += "x: " + predictedVectors[i] + "\ty: " + predictedVectors[i + 1] + "\n";
          GUI.Label(new Rect(10, 10, 500, 500), msg);
     }

     public void Train(int howManyTimes, DataType dataType)
     {
          if (dataType.Equals(DataType.Random))
          {
               Dataset[] dtst = new Dataset[howManyTimes];
               for (int i = 0; i < howManyTimes; i++)
               {
                    dtst[i] = new Dataset(inputNeuronCount, outputNeuronCount);
                    dtst[i].generateData();
                    //dtst[i].storeDataset();
                    TrainOnce(dtst[i].inputs, dtst[i].outputs);
               }
          }
     }

     public float[] Test(float[] inputs)
     {
          return net.FeedForward(inputs);
     }

     public void TrainOnce(float[] inputs, float[] expectedOutputs)
     {
          net.FeedForward(inputs);
          net.BackProp(expectedOutputs);
     }

     public enum DataType { Random, FromFile }
}

public class Dataset
{
     public float[] inputs { get; set; }
     public float[] outputs { get; set; }
     private string NNDatasetsDirectory = Directory.GetCurrentDirectory() + @"\Assets\Neural Network\datasets.txt";

     public Dataset(int inputCount, int outputCount)
     {
          inputs = new float[inputCount];
          outputs = new float[outputCount];
     }

     public void generateData()
     {
          // 4 random points
          Point[] points = new Point[4];
          for (int i = 0; i < points.Length; i++)
               points[i] = new Point();

          // we're selecting 4 points on a circle
          float r = 0.1f;
          float angle = UnityEngine.Random.Range(0, 2.0f * Mathf.PI);

          // find a random point on a circle
          points[0].x = r * Mathf.Cos(angle);
          points[0].y = r * Mathf.Sin(angle);

          // ending point is symmetric by (0,0)
          points[3].x = -points[0].x;
          points[3].y = -points[0].y;

          float min, max;
          if (UnityEngine.Random.Range(0,2) % 2 == 0)
          {
               min = angle;
               max = Mathf.PI + angle;
          }
          else
          {
               max = angle;
               min = -Mathf.PI + angle; 
          }
          
          angle = UnityEngine.Random.Range(min, max);
          points[1].x = r * Mathf.Cos(angle);
          points[1].y = r * Mathf.Sin(angle);

          angle = UnityEngine.Random.Range(min, max);
          points[2].x = r * Mathf.Cos(angle);
          points[2].y = r * Mathf.Sin(angle);

          // create multiple points on the line connecting 4 points using bezier
          // i.e (20 + 6)/2 + 1 = 14, this will give 13 vectors
          int howMany = (inputs.Length + outputs.Length) / 2 + 1;

          Point[] generatedPoints = Bezier(points, howMany);
          setDatasetFromPoints(generatedPoints);
     }

     private void setDatasetFromPoints(Point[] generatedPoints)
     {
          int i = 0;
          // clone to temporary
          float[] tmp = new float[generatedPoints.Length * 2];
          foreach (Point point in generatedPoints)
          {
               tmp[i++] = point.x;
               tmp[i++] = point.y;
          }

          // set a table of vectors based on tmp
          float[] vectors = new float[tmp.Length - 2];
          for (int k = 0; k < vectors.Length; k++)
          {
               vectors[k] = tmp[k + 2] - tmp[k];
          }

          // set inputs
          for (i = 0; i < inputs.Length; i++)
               inputs[i] = vectors[i];

          // set outputs
          for (int j = inputs.Length; j < inputs.Length + outputs.Length; j++)
               outputs[j - inputs.Length] = vectors[j];
     }

     public void storeDataset()
     {
          if (!File.Exists(NNDatasetsDirectory))
          {
               File.AppendAllText(NNDatasetsDirectory, inputs.Length.ToString() + " " + outputs.Length + Environment.NewLine);
          }
          string inLine = "";
          string outLine = "";
          foreach (float inpt in inputs)
               inLine += inpt.ToString() + " ";
          inLine += Environment.NewLine;

          foreach (float outp in outputs)
               outLine += outp.ToString() + " ";
          outLine += Environment.NewLine;
          File.AppendAllText(NNDatasetsDirectory, inLine + outLine);
     }

     public Point[] Bezier(Point[] points, int howMany)
     {
          Point[] generatedPoints = new Point[howMany];

          int i = 0;
          float increment = 1.0f / (howMany - 1);
          for (float t = 0; t <= 1.0f; t += increment)
          {
               float x = Mathf.Pow(1 - t, 3) * points[0].x + 3 * t * Mathf.Pow(1 - t, 2) * points[1].x + 3 * t * t * points[2].x + t * t * t * points[3].x;
               float y = Mathf.Pow(1 - t, 3) * points[0].y + 3 * t * Mathf.Pow(1 - t, 2) * points[1].y + 3 * t * t * points[2].y + t * t * t * points[3].y;
               generatedPoints[i++] = new Point(x, y);
          }

          generatedPoints[howMany - 1] = points[points.Length - 1];

          return generatedPoints;
     }

     public class Point
     {
          public float x { get; set; }
          public float y { get; set; }

          public Point()
          {
               x = UnityEngine.Random.Range(-1.0f, 1.0f);
               y = UnityEngine.Random.Range(-1.0f, 1.0f);
          }

          public Point(float x, float y)
          {
               this.x = x;
               this.y = y;
          }

          public Point(Point p)
          {
               this.x = p.x;
               this.y = p.y;
          }

     }
}
