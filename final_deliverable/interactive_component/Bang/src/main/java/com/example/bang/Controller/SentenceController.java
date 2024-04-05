package com.example.bang.Controller;

import com.example.bang.Request.SentenceRequest;
import com.example.bang.Response.SentenceResponse;
import com.example.bang.Service.ChatGptService;
import com.example.bang.Service.PythonExecuteService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.util.StreamUtils;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@Slf4j
public class SentenceController {

    private ObjectMapper objectMapper = new ObjectMapper();

    private static Map<Integer, String> sentiments = new HashMap<>();
    @Autowired
    private ChatGptService chatGptService;

    @Autowired
    private PythonExecuteService pythonExecuteService;

    static {
        sentiments.put(0, "negative");
        sentiments.put(1, "positive");
        sentiments.put(2, "neutral");
    }

    @PostMapping("/process")
    public SentenceResponse processSentence(@RequestBody SentenceRequest sentence) {
        if (sentence == null || StringUtils.isBlank(sentence.getSentence())) {
            return SentenceResponse.success("Please input your sentence !");
        }

        // start to use ML-Model predict
        int numVal = pythonExecuteService.predict(sentence.getSentence());
        if (numVal == -1) {
            return SentenceResponse.fail();
        }
        String userSentiment = sentiments.get(numVal);
        String jsonStr = buildJson(sentence.getSentence(), userSentiment);
        if (StringUtils.isBlank(jsonStr)) {
            return SentenceResponse.success("I would believe the sentiment of input sentence is: " + userSentiment);
        }

        // chatgpt to evaluate our prediction
        String ans = chatGptService.post(jsonStr);
        if (StringUtils.isBlank(ans)) {
            return SentenceResponse.success("Our Dean of the School of Magic is Down, but I think the sentiment of your sentence is " + userSentiment);
        }
        try {
            Map<String, Object> response = objectMapper.readValue(ans, Map.class);
            List<Map> cur = (List<Map>) response.get("choices");
            Map<String, Object> res = (Map<String, Object>) cur.get(0).get("message");
            String finalStr = (String) res.get("content");
            return SentenceResponse.success("My magic Apparition tells me the sentiment of your input sentence is "
                    + userSentiment + "." + "Our Dean of the School of Magic evalue my judgement as " + finalStr);
        }catch (Exception e) {
            return SentenceResponse.success("Our Dean of the School of Magic is Down, but I think the sentiment of your sentence is " + userSentiment);
        }
    }

    @RequestMapping("/")
    public String processSentence() throws IOException {
        Resource resource = new ClassPathResource("static/sentence.html");
        return StreamUtils.copyToString(resource.getInputStream(), StandardCharsets.UTF_8);
    }

    private String buildJson(String sentence, String sentiment) {
        try {
            String structure = "{\n" +
                    "  \"model\": \"gpt-3.5-turbo\",\n" +
                    "  \"messages\": [{\"role\": \"user\", \"content\": \"%s\"}]\n" +
                    "}";
            sentence = "My machine learning model evaluate the sentiment of sentence: " +
                    sentence + " as: " + sentiment + ". Do you think it's correct? Answer me with correct or wrong";
            String jsonStr = String.format(structure, sentence);
            return jsonStr;
        }catch (Exception e) {
            log.error("buildJson failed with errors: {}", e);
        }
        return null;
    }
}
