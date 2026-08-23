const express = require("express");

const {
    analyzeProduct,
    analyzeProductUrl
} = require("../controllers/productController");

const router = express.Router();

router.post(
    "/",
    analyzeProduct
);

router.post(
    "/url",
    analyzeProductUrl
);

module.exports = router;