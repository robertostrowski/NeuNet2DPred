using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour {

    public float speed;

	// Use this for initialization
	void Start () {
        speed = 70;
	}
	
	// Update is called once per frame
	void Update () {
        

        float throttle = Input.GetAxis("Vertical");
        float rotate = Input.GetAxis("Horizontal");
        
        GetComponent<Rigidbody>().AddForce(transform.forward * throttle * Time.deltaTime * speed * this.transform.localScale.x);
        GetComponent<Rigidbody>().AddTorque(transform.up * rotate * Time.deltaTime * speed );
    }
}
