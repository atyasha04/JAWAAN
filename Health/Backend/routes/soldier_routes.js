const router=require("express").Router();
const Soldier=require('../models/soldier')

router.post("/Create_Soldier",async(req,res)=>{
    try{
    const request_body=req.body
    const newUser=new Soldier(request_body)
        await newUser.save()
        return res.status(200).json({message:"SOLDIER SIGNUP SUCCESSFULLY"})


}
catch(error){
    console.error("Sign-up error:", error);
    res.status(500).json({message:"INTERNAL SERVER ERROR"})
}
})

router.get("/get-all-Soldiers", async (req, res) => {
    try {
      const Data = await Soldier.find()
        // .sort({ createdAt: -1 });
      return res.json({
        status: "Success",
        data: Data,
      });
    } catch (error) {
      return res.status(500).json({ message: "Internal Server Error", error });
    }
  });

  router.get('/search', async (req, res) => {
    try {
        const {
            name,
            armyId,
            armyRank,
            armyBattalion,
            healthStatus,
            heart_rate_min,
            heart_rate_max,
            blood_pressure_min,
            blood_pressure_max,
            oxygen_saturation_min,
            oxygen_saturation_max,
            stress_level_min,
            stress_level_max,
            steps_min,
            steps_max,
            sleep_hours_min,
            sleep_hours_max,
            activity_level_min,
            activity_level_max
        } = req.query;

        // Build the search query
        const query = {};

        if (name) query.name = { $regex: name, $options: 'i' }; // Case-insensitive regex
        if (armyId) query.armyId = armyId;
        if (armyRank) query.armyRank = armyRank;
        if (armyBattalion) query.armyBattalion = armyBattalion;
        if (healthStatus) query.healthStatus = healthStatus;

        // Add range filters for numeric fields
        if (heart_rate_min || heart_rate_max) {
            query.heart_rate = {};
            if (heart_rate_min) query.heart_rate.$gte = Number(heart_rate_min);
            if (heart_rate_max) query.heart_rate.$lte = Number(heart_rate_max);
        }

        if (blood_pressure_min || blood_pressure_max) {
            query.blood_pressure = {};
            if (blood_pressure_min) query.blood_pressure.$gte = Number(blood_pressure_min);
            if (blood_pressure_max) query.blood_pressure.$lte = Number(blood_pressure_max);
        }

        if (oxygen_saturation_min || oxygen_saturation_max) {
            query.oxygen_saturation = {};
            if (oxygen_saturation_min) query.oxygen_saturation.$gte = Number(oxygen_saturation_min);
            if (oxygen_saturation_max) query.oxygen_saturation.$lte = Number(oxygen_saturation_max);
        }

        if (stress_level_min || stress_level_max) {
            query.stress_level = {};
            if (stress_level_min) query.stress_level.$gte = Number(stress_level_min);
            if (stress_level_max) query.stress_level.$lte = Number(stress_level_max);
        }

        if (steps_min || steps_max) {
            query.steps = {};
            if (steps_min) query.steps.$gte = Number(steps_min);
            if (steps_max) query.steps.$lte = Number(steps_max);
        }

        if (sleep_hours_min || sleep_hours_max) {
            query.sleep_hours = {};
            if (sleep_hours_min) query.sleep_hours.$gte = Number(sleep_hours_min);
            if (sleep_hours_max) query.sleep_hours.$lte = Number(sleep_hours_max);
        }

        if (activity_level_min || activity_level_max) {
            query.activity_level = {};
            if (activity_level_min) query.activity_level.$gte = Number(activity_level_min);
            if (activity_level_max) query.activity_level.$lte = Number(activity_level_max);
        }

        // Execute the query
        const soldiers = await Soldier.find(query);
        res.status(200).json(soldiers);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal server error' });
    }
});





module.exports=router;