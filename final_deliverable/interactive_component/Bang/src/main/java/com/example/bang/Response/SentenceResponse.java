package com.example.bang.Response;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Data
public class SentenceResponse {

    private int code;
    private String data;

    private String msg;

    private SentenceResponse() {};

    private SentenceResponse(int code, String data, String msg) {
        this.code = code;
        this.data = data;
        this.msg = msg;
    }

    public static SentenceResponse success() {
        return new SentenceResponse(200, null, null);
    }

    public static SentenceResponse success(String data) {
        return new SentenceResponse(200, data, "");
    }

    public static SentenceResponse fail() {
        return new SentenceResponse(500, null, "Internal Errors");
    }

    public static SentenceResponse fail(String data) {
        return new SentenceResponse(500, data, "Internal Errors");
    }
}
