const express = require("express");

const {
    analyzeUrl
} = require("../controllers/urlController");


const router = express.Router();


router.post(
    "/",
    analyzeUrl
);


module.exports = router;




// const express = require("express");
// const axios = require("axios");

// const router = express.Router();

// router.post("/", async (req, res) => {
//     try {
//         const { url } = req.body;

//         if (!url || !url.trim()) {
//             return res.status(400).json({
//                 error: "URL cannot be empty"
//             });
//         }

//         const response = await axios.post(
//             `${process.env.FASTAPI_URL}/api/v1/predict/url`,
//             {
//                 url: url
//             }
//         );

//         res.json(response.data);

//     } catch (error) {

//         console.error(
//             "Phishing prediction error:",
//             error.message
//         );

//         res.status(500).json({
//             error: "Failed to analyze URL"
//         });
//     }
// });

// module.exports = router;