
const mongoose = require('mongoose');

const item = new mongoose.Schema({
  itemName: {
    type: String,
    required: true,
  },
  itemCategory: {
    type: String,
    enum: ['Grains','Vegetables','Fruits','Dairy','Meat','Snacks','Others','Fish','Poultry','Spices','Weapons'],
    required: true,
  },
   imageUrl: {
     type: String, 
      required: true 
    },
  quantityInStock: {
    type: Number,
    required: true,
    min: 0,
  },
  warehouseExactLocation: { 
    shelf: { type: String, required: true },
    room: { type: String, required: true },
    floor: { type: Number, required: true },
    rackNumber: { type: Number, required: true },
    placeInRack: { type: String, required: true }
 },
  unit: {
    type: String,
    enum: ['kg', 'liters', 'units'],
    required: true,
  },
  consumptionPerDayPerPerson: {
    type: Number,
    required: true,
    min: 0,
  },
  last14DaysConsumptionperPerson: {
    type: [Number], // Array of numbers representing daily consumption for the last 14 days
    validate: {
      validator: function (value) {
        return value.length <= 14;
      },
      message: 'Consumption history should not exceed 14 days.',
    },
  },
  lastUpdated: {
    type: Date,
    default: Date.now,
  },
  reorderThreshold: {
    type: Number,
    required: true,
    min: 0
  },

supplierName: {
      type: String,
      required: true,
      trim: true,
    },
contactNumber: {
      type: String,
      required: true,
      match: /^[0-9]{10}$/,
    },
    expiryDate: {
        type: Date,
        required:true
      
    }
      
  
      
});

item.virtual('lowStockStatus').get(function () {
    return this.quantityInStock < this.reorderThreshold;
  });
item.virtual('isExpired').get(function () {
     return this.expiryDate < new Date(); 
}); 
      item.set('toJSON', { virtuals: true }); 
      item.set('toObject', { virtuals: true });

const Item= mongoose.model('items', item);

module.exports = Item;

