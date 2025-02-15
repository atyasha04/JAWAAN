const express = require('express');
const router = express.Router();
const Item = require('../models/item'); // Path to your Item model

// 1. Create a new item
router.post('/items', async (req, res) => {
  try {
    const item = new Item(req.body);
    const data=await item.save();
    res.status(201).json(data);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// 2. Get all items
router.get('/items', async (req, res) => {
  try {
    const items = await Item.find().limit(20);
    res.status(200).json(items);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// 3. Get a single item by ID
router.get('/items/:id', async (req, res) => {
  try {
    const item = await Item.findById(req.params.id);
    if (!item) return res.status(404).json({ message: 'Item not found' });
    res.status(200).json(item);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// 4. Update an item by ID
router.put('/items/:id', async (req, res) => {
  try {
    const item = await Item.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
      runValidators: true,
    });
    if (!item) return res.status(404).json({ message: 'Item not found' });
    res.status(200).json(item);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// 5. Delete an item by ID
router.delete('/items/:id', async (req, res) => {
  try {
    const item = await Item.findByIdAndDelete(req.params.id);
    if (!item) return res.status(404).json({ message: 'Item not found' });
    res.status(200).json({ message: 'Item deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});
// 6. Search items by various fields with expired and low stock filters
router.get('/search/items', async (req, res) => {
    const { itemName, itemCategory, supplierName, expiryDate, lowStock, expired } = req.query;
  
    const query = {};
  
    // Filter by item name (case-insensitive)
    if (itemName) {
      query.itemName = { $regex: itemName, $options: 'i' };
    }
  
    // Filter by item category
    if (itemCategory) {
      query.itemCategory = itemCategory;
    }
  
    // Filter by supplier name (case-insensitive)
    if (supplierName) {
      query.supplierName = { $regex: supplierName, $options: 'i' };
    }
  
    // Filter by expiry date (items expiring before the given date)
    if (expiryDate) {
      query.expiryDate = { $lte: new Date(expiryDate) };
    }
  
    // Filter by low stock (items where quantityInStock is below reorderThreshold)
    if (lowStock === 'true') {
      query.$expr = { $lt: ["$quantityInStock", "$reorderThreshold"] }; // This checks if stock is less than reorder threshold
    }
  
    // Filter by expired items (items whose expiry date has passed)
    if (expired === 'true') {
      query.expiryDate = { $lt: new Date() }; // Items that have expired
    }
  
    try {
      const items = await Item.find(query).limit(20);
      res.status(200).json(items);
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
  });

  module.exports = router;

  