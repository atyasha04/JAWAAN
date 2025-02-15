const express=require("express")
const app=express();

require("./connection/connection.js")
const inventory=require("./routes/inventory.js")
app.use(express.json());
const cors = require('cors');
app.use(cors());

//const soldier=require("./routes/soldier_routes.js")
// const book=require("./routes/book.js")
// const favourite=require("./routes/favourite.js")
// const cart=require("./routes/cart.js")
// const order=require("./routes/order.js")
//user using json format





//routes
app.use("/api/v1",inventory)
// app.use("/api/v1",book)
// app.use("/api/v1",favourite)
// app.use("/api/v1",cart)
// app.use("/api/v1",order)






//creating the port of the app
app.listen(1000,()=>{
    console.log(`SERVER STARTED at port 1000`)
})