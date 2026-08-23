require("dotenv").config();

const express = require("express");
const cors = require("cors");

const healthRoute = require("./routes/health");
const reviewRoute = require("./routes/review");
const urlRoute = require("./routes/url");
const productRoute = require("./routes/product");

const app = express();

app.use(cors());
app.use(express.json());

app.use("/health", healthRoute);
app.use("/api/reviews", reviewRoute);
app.use("/api/urls", urlRoute);
app.use("/api/products", productRoute);

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`);
});