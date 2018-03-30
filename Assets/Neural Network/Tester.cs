using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Tester : MonoBehaviour
{
    // To żeby móc się dostać do tych obiektów
    private Transform playerTransform;
    private Transform ball1;
    private Transform ball2;
    private Transform ball3;

    // DO kontroli liczby iteracji
    private int iterations;


    // tablica trzymająca wejścia i wyjścia sieci
    float[] tab = new float[28];
    float[] temp = new float[6];
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

        iterations = 0;
        


        // żeby w tablicy nie było śmieci
        for (int i = 0; i < 28; i++)
            tab[i] = 0.0f;




        //każda wartość w nawiasie { } to liczba komórek w danej warstwie, czyli mamy tu 20 na wejście, 13 w warstwie ukrytej i 6 na wyjście
        // - learning rate - określa wpływ nowych wejść na dotychczasowe wagi, im większy tym szybsze zmiany
        net = new NeuralNet(new int[] { 20, 13, 13, 6 }, 0.00333f);





    }

    // Komunikat na ekran, argument musi być stringiem więc jak masz liczbę do daj X.toString();
    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 100, 20), temp[0].ToString());
    }
    // Update is called once per frame
    void Update()
    {
        // zrobione żeby całość wykonywała się co X iteracji
        if (iterations % 10 == 0)
        {

            //Nauka - bierzemy pierwsze 20 różnic odległości i jako oczekiwany wynik przekazujemy 3 następne
                for (int i = 0; i < 10; i++)
                {
                   temp  = net.FeedForward(new float[] { tab[2] - tab[0], tab[3] - tab[1], tab[4] - tab[2], tab[5] - tab[3], tab[6] - tab[4],
                                              tab[7] - tab[5], tab[8] - tab[6], tab[9] - tab[7], tab[10] - tab[8], tab[11] - tab[9],
                                              tab[12] - tab[10], tab[13] - tab[11], tab[14] - tab[12], tab[15] - tab[13], tab[16] - tab[14],
                                              tab[17] - tab[15], tab[18] - tab[16], tab[19] - tab[17], tab[20] - tab[18], tab[21] - tab[19],
        });
                    net.BackProp(new float[] { tab[22] - tab[20], tab[23] - tab[21], tab[24] - tab[22], tab[25] - tab[23], tab[26] - tab[24], tab[27] - tab[25] });


                }

            //Zmiana - przesuwanie wartości w tabeli
            for (int i = 2; i < 28; i++)
            {
                tab[i - 2] = tab[i];
            }
            tab[26] = playerTransform.position.x;
            tab[27] = playerTransform.position.z;


            // Przypisanie wyjść danej iteracji do tabeli pomocniczej
            
            
         

            // Wyniki nauki przekazujemy do współrzędnych punktów 1,2,3 - 20 argumentów jako wejścia a wyjścia poprzez indeksy 0-5
            // 1 Punkt
            ball1.position = new Vector3(
                                                //x
                                              temp[0] + playerTransform.position.x, 
                                                 //y
                                                 playerTransform.position.y,
                                                 //z
                                                  temp[1] + playerTransform.position.z
                                        );

            // 2 Punkt
            ball2.position = new Vector3(
                                                 //x
                                             temp[0] + ball1.position.x,
                                                  //y
                                                  playerTransform.position.y,
                                                    //z
                                             temp[1] + ball1.position.z
                                         );

            // 3 Punkt
            ball3.position = new Vector3(
                                                //x
                                               temp[0] + ball2.position.x,
                                                 //y
                                                 playerTransform.position.y,
                                                   //z
                                                 temp[1] + ball2.position.z
                                        );


            // Punkty stworzą krzywą, która wskaże jak leciał statek przed chwilą
            //ball1.position = new Vector3(tab[18], playerTransform.position.y,tab[19]);
            //ball2.position = new Vector3(tab[20], playerTransform.position.y, tab[21]);
            //ball3.position = new Vector3(tab[22], playerTransform.position.y, tab[23]);

        }

        iterations++;
        OnGUI();
    }
}
