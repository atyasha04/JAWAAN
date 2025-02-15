const mongoose=require("mongoose")

const soldier=new mongoose.Schema({
    name:{
        type:String,
        required:true
    },
    armyId:{
        type:String,
        required:true

    },
    armyRank:{
        type:String,
        required:true
    },
    armyBattalion:{
        type:String,
        required:true
    },
    healthStatus:{
        type:String,
        enum:["normal","alert"]
    },
    heart_rate:{
        type:Number,
        required:true

    },
    blood_pressure:{
        type:Number,
        required:true
    },
    oxygen_saturation:{
        type:Number,
        required:true

    },
    stress_level:{
        type:Number,
        required:true
    },
    steps:{
        type:Number,
        required:true
    },
    sleep_hours:{
        type:Number,
        required:true
    },
    activity_level:{
        type:Number,
        required:true
    }

},{timestamps:true})
module.exports=mongoose.model('soldiers',soldier)