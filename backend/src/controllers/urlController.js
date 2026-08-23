const {
    predictUrl
} = require("../services/mlService");


async function analyzeUrl(req, res) {

    try {

        const { url } = req.body;

        if (!url || !url.trim()) {

            return res.status(400).json({
                error: "URL cannot be empty"
            });

        }

        const result = await predictUrl(url);

        return res.json(result);

    } catch (error) {

        console.error(
            "Phishing prediction error:",
            error.message
        );

        return res.status(500).json({
            error: "Failed to analyze URL"
        });
    }
}


module.exports = {
    analyzeUrl
};