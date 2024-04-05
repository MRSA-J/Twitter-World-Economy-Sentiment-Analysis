package com.example.bang.Service;

import lombok.extern.slf4j.Slf4j;
import okhttp3.MediaType;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import okhttp3.*;

import java.io.IOException;

@Service
@Slf4j
public class ChatGptService {

    private final static String APIKEY = "Bearer " + "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";

    private final static String APIURL ="https://api.openai.com/v1/chat/completions";

    OkHttpClient client = new OkHttpClient();

    public String post(String json){
        RequestBody body = RequestBody.create(
                MediaType.parse("application/json; charset=utf-8"), json);

        Request request = new Request.Builder()
                .url(APIURL)
                .post(body)
                .addHeader("Authorization", APIKEY)
                .build();

        try (Response response = client.newCall(request).execute()) {
            System.out.print(response.body());
            return response.body().string();
        }catch (Exception e) {
            log.error("ChatGptService failed with errors: {}", e);
        }
        return null;
    }
}
