using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{

     public float speed;

     // Use this for initialization
     void Start() { }

     // Update is called once per frame
     void Update()
     {
          // get input
          float rotate = Input.GetAxis("Horizontal");

          transform.position += transform.forward * speed * Time.deltaTime;
          var rotateVector = new Vector3(0, rotate, 0);
          transform.Rotate(rotateVector);
     }
}
