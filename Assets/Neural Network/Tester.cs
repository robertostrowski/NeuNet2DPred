﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

public class Tester : MonoBehaviour
{
    // To żeby móc się dostać do tych obiektów
    private Transform playerTransform;
    private Transform ball1;
    private Transform ball2;
    private Transform ball3;

    public int learningIteriations = 1;
    public float learningRate = 0.00333f;
    public bool generateNewDatasets;
    public int numberOfTrainingDatasets;
    public int numberOfTestingDatasets;

    private readonly string trainingDatasetsPath = Directory.GetCurrentDirectory() + @"\Assets\Neural Network\Data\trainingDatasets.csv";
    private readonly string testingDatasetsPath = Directory.GetCurrentDirectory() + @"\Assets\Neural Network\Data\testingDatasets.csv";
    private string errorsPath = Directory.GetCurrentDirectory() + @"\Assets\Neural Network\Data";

    public static int inputNeuronCount = 20;
    public static int outputNeuronCount = 6;

    private float distPerIter;

    float[] lastPositions = new float[inputNeuronCount];
    float[] predictedVectors = new float[outputNeuronCount];
    NeuralNet net;

    float iterations = 0;

    // Use this for initialization
    void Start()
    {
        playerTransform = GameObject.Find("Statek").transform;
        ball1 = GameObject.Find("ball1").transform;
        ball2 = GameObject.Find("ball2").transform;
        ball3 = GameObject.Find("ball3").transform;

        BuildErrorsFileName();

        if (generateNewDatasets)
        {
            Dataset[] datasets = GenerateDatasets(numberOfTrainingDatasets);
            StoreDatasets(datasets, trainingDatasetsPath);

            datasets = GenerateDatasets(numberOfTestingDatasets);
            StoreDatasets(datasets, testingDatasetsPath);
        }

        net = new NeuralNet(new int[] { inputNeuronCount, 13, outputNeuronCount }, learningRate);
        Dataset[] testDatasets = LoadDatasets(testingDatasetsPath);
        Dataset[] trainDatasets = LoadDatasets(trainingDatasetsPath);

        // do csv error, wykres, stałe zestawy
        Train(trainDatasets, trainDatasets.Length, learningIteriations);
        //Test(testDatasets);

    }

    private void BuildErrorsFileName()
    {
        string suffix = "";
        suffix += "it " + learningIteriations.ToString();
        suffix += " lr " + learningRate.ToString("n5");
        suffix += ".csv";
        errorsPath = Path.Combine(errorsPath, suffix);
    }

    private void StoreDatasets(Dataset[] datasets, string path)
    {
        StringBuilder sb = new StringBuilder();
        foreach (Dataset ds in datasets)
        {
            sb.Append(string.Join(",", ds.inputs.Select(v => v.ToString()).ToArray()));
            sb.Append(Environment.NewLine);
            sb.Append(string.Join(",", ds.outputs.Select(v => v.ToString()).ToArray()));
            sb.Append(Environment.NewLine);
        }
        File.WriteAllBytes(path, Encoding.ASCII.GetBytes(sb.ToString()));
    }

    private Dataset[] LoadDatasets(string path)
    {
        string[] s = File.ReadAllLines(path);
        Dataset[] ds = new Dataset[s.Length / 2];
        for (int i = 0; i < ds.Length; i++)
        {
            float[] inputs = s[2 * i].Split(',').Select(v => float.Parse(v)).ToArray();
            float[] outputs = s[2 * i + 1].Split(',').Select(v => float.Parse(v)).ToArray();
            ds[i] = new Dataset(inputs, outputs);
        }
        return ds;
    }

    private Dataset[] GenerateDatasets(int howMany)
    {
        Dataset[] ds = new Dataset[howMany];
        for (int i = 0; i < howMany; i++)
        {
            ds[i] = new Dataset(inputNeuronCount, outputNeuronCount);
            ds[i].generateData();
        }
        return ds;
    }

    // Update is called once per frame
    void Update()
    {
        if (iterations++ % 10 != 0)
            return;

        // new position
        float newX = playerTransform.position.x;
        float newY = playerTransform.position.z;

        // nn is taking vectors for input
        float[] lastPosVectors = new float[lastPositions.Length];
        // fill the table with vectors except for 2 last elements
        for (int i = 0; i < lastPosVectors.Length - 2; i++)
        {
            lastPosVectors[i] = lastPositions[i + 2] - lastPositions[i];
        }
        // fill the last 2 elements
        lastPosVectors[lastPosVectors.Length - 2] = newX - lastPositions[lastPosVectors.Length - 2];
        lastPosVectors[lastPosVectors.Length - 1] = newY - lastPositions[lastPosVectors.Length - 1];

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


        updateDistancePerIter(lastPosVectors[lastPosVectors.Length - 1], lastPosVectors[lastPosVectors.Length - 2]);
    }

    private float[] Test(float[] lastPosVectors)
    {
        return net.FeedForward(lastPosVectors);
    }

    void dropNewGreenBall(float x, float y)
    {
        var ball = Instantiate(ball1);
        ball.transform.position.Set(x, 0, y);
    }

    void dropNewYellowBall(float x, float y)
    {
        var ball = Instantiate(ball2);
        ball.transform.position.Set(x, 0, y);
    }

    private void updateDistancePerIter(float vx, float vy)
    {
        distPerIter = Mathf.Sqrt(Mathf.Pow(vx, 2) + Mathf.Pow(vy, 2));
    }

    public void Train(Dataset[] datasets, int howManyDatasets, int noOfIterations)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < howManyDatasets; i++)
        {
            float[] errorVector = net.getErrorVector();
            String s = string.Join(",", errorVector.Select(v => v.ToString()).ToArray());
            sb.Append(s);
            sb.Append(Environment.NewLine);

            for (int j = 0; j < noOfIterations; j++)
                TrainOnce(datasets[i].inputs, datasets[i].outputs);
        }
        File.WriteAllBytes(errorsPath, Encoding.ASCII.GetBytes(sb.ToString()));
    }

    public void Test(Dataset[] datasets)
    {
        StringBuilder sb = new StringBuilder();
        foreach (Dataset ds in datasets)
        {
            float[] outputs = net.FeedForward(ds.inputs);
            float[] errorVector = ds.outputs;
            for (int i = 0; i < errorVector.Length; i++)
            {
                errorVector[i] -= outputs[i];
            }

            String s = string.Join(",", errorVector.Select(v => v.ToString()).ToArray());
            sb.Append(s);
            sb.Append(Environment.NewLine);
        }
        File.WriteAllBytes(errorsPath, Encoding.ASCII.GetBytes(sb.ToString()));
    }

    public void TrainOnce(float[] inputs, float[] expectedOutputs)
    {
        net.FeedForward(inputs);
        net.BackProp(expectedOutputs);
    }
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

    public Dataset(float[] inputs, float[] outputs)
    {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public void generateData()
    {
        // for 13 vectros u need 14 points
        Point[] points = new Point[(inputs.Length + outputs.Length) / 2 + 1];

        //  select a random curve on a circle, pick points from that curve

        // angleStart - starting point of a curve on a circle, at that angle
        float angleStart, angleEnd, randomAngle;

        angleStart = UnityEngine.Random.Range(0, Mathf.PI * 2.0f);

        randomAngle = UnityEngine.Random.Range(Mathf.PI / 2, Mathf.PI * 3 / 4);

        angleEnd = UnityEngine.Random.Range(0, 2) % 2 == 0 ? angleStart + randomAngle : angleStart - randomAngle;

        float angleInc = (angleStart - angleEnd) / (inputs.Length + outputs.Length + 1);

        float angle = angleStart;

        for (int i = 0; i < points.Length; i++)
        {
            float x, y;
            // assume r = 1
            x = Mathf.Cos(angle);
            y = Mathf.Sin(angle);

            points[i] = new Point(x, y);

            angle += angleInc;
        }
        // important, scaling 
        // take 2 points, calculate distance, scale so the distance is ~0.2
        float scale = Mathf.Sqrt((points[1].x - points[0].x) * (points[1].x - points[0].x) + (points[1].y - points[0].y) * (points[1].y - points[0].y));

        foreach (Point point in points)
        {
            point.x = point.x / scale;
            point.y = point.y / scale;
        }
        setDatasetFromPoints(points);
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
