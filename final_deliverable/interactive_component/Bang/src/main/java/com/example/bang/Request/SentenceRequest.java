package com.example.bang.Request;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Data
public class SentenceRequest {
    private String sentence;

    public SentenceRequest() {

    }

    public SentenceRequest(String sentence) {
        this.sentence = sentence;
    }
}
