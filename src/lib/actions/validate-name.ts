"use server";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

export const validateName = async (name: string) => {
  try {
    const geminiAPIKey = process.env.GEMINI_API_KEY;

    const messages = [
      new SystemMessage(
        "Your job is to extract the name from the user's message which can be a sentence, word or it's name too, but if the name is invalid or not a human name then return the boolean value 'false' only."
      ),
      new HumanMessage(name),
    ];

    const llm = new ChatGoogleGenerativeAI({
      model: "gemini-pro",
      apiKey: geminiAPIKey,
      temperature: 0.1,
    });

    const isCorrect = await llm.invoke(messages);

    return { isCorrect: isCorrect.content };
  } catch (error: any) {
    return {
      error: error.message,
    };
  }
};
