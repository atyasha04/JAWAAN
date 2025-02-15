const express=require("express")
const app=express();

require("./connection/conn")
const soldier=require("./routes/soldier_routes.js")
// const book=require("./routes/book.js")
// const favourite=require("./routes/favourite.js")
// const cart=require("./routes/cart.js")
// const order=require("./routes/order.js")
//user using json format
const cors = require('cors');

app.use(express.json());

app.use(cors());


//routes
app.use("/api/v1",soldier)
// app.use("/api/v1",book)
// app.use("/api/v1",favourite)
// app.use("/api/v1",cart)
// app.use("/api/v1",order)






//creating the port of the app
app.listen(1000,()=>{
    console.log(`SERVER STARTED at port 1000`)
})