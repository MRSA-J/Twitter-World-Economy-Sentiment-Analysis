package com.example.bang.Service;

import jnr.ffi.annotations.In;
import lombok.extern.slf4j.Slf4j;
import org.python.core.PyFunction;
import org.python.core.PyString;
import org.python.util.PythonInterpreter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import org.python.core.PyObject;


import java.io.*;


@Service
@Slf4j
public class PythonExecuteService {

    private PythonInterpreter interpreter = new PythonInterpreter();

    private  String pythonScriptPath = "/Users/sunzhenhao/IdeaProjects/Bang/src/main/resources/static/MultinomialNB.py";
    private  String output =  "static/output.txt";

    public int predict(String sentence) {
        Integer val;
        try {
            ProcessBuilder pb = new ProcessBuilder("python3", pythonScriptPath);

            // Merge the error stream and the standard output stream
            pb.redirectErrorStream(true);
            Process p = pb.start();

            // Write to the process's input stream
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));
            out.write(sentence);
            out.newLine();
            out.flush();

            // Read from the process's output stream
            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String ret = in.readLine();
            val = Integer.valueOf(ret);
        }catch (Exception e) {
            return -1;
        }
        return val;
    }
}
