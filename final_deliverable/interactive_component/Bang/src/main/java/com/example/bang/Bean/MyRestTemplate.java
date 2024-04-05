package com.example.bang.Bean;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;


@Configuration
public class MyRestTemplate {

    @Bean
    public RestTemplate RestTemplateFactory() {
        RestTemplate restTemplate = new RestTemplate();
        return restTemplate;
    }
}
