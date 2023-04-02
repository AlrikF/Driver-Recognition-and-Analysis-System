import axios from 'axios'
import './Anne.css';
import React, { Component } from 'react'

var DOMAIN = '' //http://your-domain => http://localhost:5000
var SRC = "https://drive.google.com/uc?id="

export default class Anne extends Component {

  constructor(props) {
    super(props)
    
    this.state = ( {
      res:[],
      image_ids:[],
      classes:[],
      counter:0,
      new_class_name:"",
    })
  }

  componentDidMount () {
    axios.get(DOMAIN+'/getimages')
    .then(Response =>{
        console.log(Response)
        this.setState ({res:Response.data})
        var i=0;
        var image_ids=[];
        console.log(this.state.res)
      for(i=0;i<this.state.res.length;i++)
      {
        image_ids.push(this.state.res[i]["id"])
      }
      this.setState({image_ids:image_ids})
      console.log(this.state.image_ids)
    })
    .catch(error =>{
        console.log(error)
    })

    axios.get(DOMAIN+'/getclasses')
    .then(Response=>{
      console.log(Response)
      var i=0;
      var classes=[]
      for(var _class in Response.data)
      {
          classes.push(_class)
      }
      this.setState({classes:classes})
      console.log(classes)
    })
    .catch(error=>{
      console.log(error)
    })

    
}

addNewClass=()=>{
   
  console.log("in addNewClass")
  console.log(this.state.new)
  console.log("so finally the selected is "+this.state.new_class_name)

  axios.post(DOMAIN+'/setannotation/'+this.state.image_ids[this.state.counter]+"/"+this.state.new_class_name+"/1")
    .then(Response=>{console.log(Response)})
    .catch(error=>{console.log(error)})

 this.setState({counter:this.state.counter+1}); 
}

addtoExistingClass=()=>{
   
  console.log("in addNewClass")
  console.log(this.state.new)
  console.log("so finally the selected is "+this.state.new_class_name)

  axios.post(DOMAIN+'/setannotation/'+this.state.image_ids[this.state.counter]+"/"+this.state.new_class_name+"/0")
    .then(Response=>{console.log(Response)})
    .catch(error=>{console.log(error)})

 this.setState({counter:this.state.counter+1}); 
}

changeHandler = (event) =>{   this.setState({[event.target.name]:event.target.value}) };

handleSelectChange = (event) => { this.setState({new_class_name:event.target.value}) };

  render() {

    const {new_class_name,classes}=this.state
    var imageSource=SRC+this.state.image_ids[this.state.counter]


    if(this.state.counter<this.state.image_ids.length)

    return(
      
      <div className="App">

        <br></br>
        <br></br>

        <h1>ANNE</h1>

        <br></br>
        <br></br>

        <img src={imageSource} alt="driver"></img><br></br>

        <br></br>
        <br></br>

        <label>Type in a new class name : </label>

        <br></br>

        <input  name="new_class_name" value={new_class_name} onChange={this.changeHandler} type="text"></input>
        <button  onClick={this.addNewClass}>Add New Class</button>

        <br></br>
        <br></br>

        <label> OR </label>

        <br></br>
        <br></br>

        <label> Insert into existing class </label>
        <br></br>
        <br></br>

        <select 
        onChange={this.handleSelectChange} >
        <option disabled selected value> -- select an option -- </option>
        {
          classes.length ?
            classes.map(_class=>
              <option key={_class.id} value={_class}>{_class}
              </option>) 
            :null
        }
        </select>
        <button  onClick={this.addtoExistingClass}>Add to Existing Class</button>
        <br></br>
        <br></br>
        <br></br>
        <br></br>
      </div>
    )



    return (
      <div className="App">
        <h1 >All images Annotated</h1>
      </div>
    )
  }
}

