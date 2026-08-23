const express = require("express");

const {
    analyzeReview
} = require("../controllers/reviewController");


const router = express.Router();


router.post(
    "/",
    analyzeReview
);


module.exports = router;

// const express = require("express");
// const axios = require("axios");

// const router = express.Router();

// router.post("/", async (req, res) => {
//     try {
//         const { review } = req.body;

//         if (!review || !review.trim()) {
//             return res.status(400).json({
//                 error: "Review cannot be empty"
//             });
//         }

//         const response = await axios.post(
//             `${process.env.FASTAPI_URL}/api/v1/predict/review`,
//             {
//                 review: review
//             }
//         );

//         res.json(response.data);

//     } catch (error) {

//         console.error(
//             "Fake review prediction error:",
//             error.message
//         );

//         res.status(500).json({
//             error: "Failed to analyze review"
//         });
//     }
// });

// module.exports = router;