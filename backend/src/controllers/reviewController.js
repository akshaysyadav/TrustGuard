const {
    predictReview
} = require("../services/mlService");


async function analyzeReview(req, res) {

    try {

        const { review } = req.body;

        if (!review || !review.trim()) {

            return res.status(400).json({
                error: "Review cannot be empty"
            });

        }

        const result = await predictReview(review);

        return res.json(result);

    } catch (error) {

        console.error(
            "Review prediction error:",
            error.message
        );

        return res.status(500).json({
            error: "Failed to analyze review"
        });
    }
}


module.exports = {
    analyzeReview
};