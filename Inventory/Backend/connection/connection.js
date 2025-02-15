const mongoose=require('mongoose')

const conn = async () =>{
    try{
        await mongoose.connect("mongodb+srv://subhanjansaha2003:7IsxUHLhp1l3mIvl@bookuser.9hnso.mongodb.net/Project0")
        console.log("connected to the database")
    }
    catch(error){
        console.log(error)

    }



};
conn();