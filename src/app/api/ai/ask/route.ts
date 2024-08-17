import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { v4 as uuidv4 } from "uuid";

export const maxDuration = 60;

export const POST = async (req: NextRequest) => {
  try {
    const { message } = await req.json();

    const geminiAPIKey = process.env.GEMINI_API_KEY;
    const pineconeAPIKey = process.env.PINECONE_API_KEY;
    const pineconeIndexName = process.env.PINECONE_INDEX_NAME;

    if (!geminiAPIKey) {
      return NextResponse.json(
        { error: "Gemini API key is not set" },
        { status: 401 }
      );
    }

    if (!pineconeAPIKey) {
      return NextResponse.json(
        { error: "Pinecone API key is not set" },
        { status: 401 }
      );
    }

    if (!pineconeIndexName) {
      return NextResponse.json(
        { error: "Pinecone Index Name is not set" },
        { status: 401 }
      );
    }

    const pinecone = new Pinecone({
      apiKey: pineconeAPIKey,
    });

    const pineconeIndex = pinecone.Index(pineconeIndexName);

    const embedModel = new GoogleGenerativeAIEmbeddings({
      model: "embedding-001", // 768 dimensions
      apiKey: process.env.GEMINI_API_KEY,
    });

    const chatModel = new ChatGoogleGenerativeAI({
      temperature: 0,
      apiKey: process.env.GEMINI_API_KEY,
      modelName: "gemini-pro",
    });

    const vectorStore = await PineconeStore.fromExistingIndex(embedModel, {
      pineconeIndex,
    });

    const qaSystemPrompt = `You are Support.AI, a chatbot designed to assist customers with their support needs. Use the following pieces of context to answer the question:
    \n\n\n
    {context}
    \n\n\n
    Guidelines:
    \n\n
    
    1. Identify and respond to conversational lines such as "Hi, how are you?", "Thanks a lot for the help", "Have a nice day" etc. Exchange pleasantries and respond in a friendly manner, as if you are a customer support agent, not a robot. However, do not respond to any unnecessary questions that are not related to customer support.
    \n\n
    2. Ensure you thoroughly understand the question. Look for a similar question in the provided context and use that to form your response.
    \n\n
    3. If the question is completely different from the context, if the answer is not available in the context, or if the question is not related to customer support, respond with: "Please ask a relevant question related to customer support."
    \n\n
    4. If the user's question is unclear or doesn't exactly match the context, find the most similar question within the context and answer that.
    `;

    const qaPromptTemplate = ChatPromptTemplate.fromMessages([
      ["system", qaSystemPrompt],
      ["human", "{input}"],
    ]);

    const questionAnswerChain = await createStuffDocumentsChain({
      llm: chatModel,
      prompt: qaPromptTemplate,
    });
    const retriever = vectorStore.asRetriever();

    const retrievalChain = await createRetrievalChain({
      combineDocsChain: questionAnswerChain,
      retriever,
    });

    const response = await retrievalChain.invoke({
      input: message,
    });

    // const transformedStream = new ReadableStream({
    //   async start(controller) {
    //     const reader = answerStream.getReader();
    //     while (true) {
    //       const { done, value } = await reader.read();
    //       if (done) break;

    //       // Transform the value to match the expected structure
    //       controller.enqueue({
    //         content: value?.content || JSON.stringify(value),
    //       });
    //     }
    //     controller.close();
    //   },
    // });

    return NextResponse.json(
      {
        message: {
          id: uuidv4(),
          type: "ai",
          content: response.answer,
        },
      },
      { status: 200 }
    );
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error }, { status: 500 });
  }
};
