const mongoose=require('mongoose')

const conn = async () =>{
    try{
        
        console.log("connected to the database")
    }
    catch(error){
        console.log(error)

    }



};
conn();
