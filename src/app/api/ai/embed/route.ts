import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { NextResponse } from "next/server";

export const maxDuration = 60;

export const POST = async () => {
  try {
    const geminiAPIKey = process.env.GEMINI_API_KEY;
    const pineconeAPIKey = process.env.PINECONE_API_KEY;
    const pineconeIndexName = process.env.PINECONE_INDEX_NAME;

    if (!geminiAPIKey) {
      return NextResponse.json(
        { error: "Gemini API key isn't set" },
        { status: 401 }
      );
    }

    if (!pineconeAPIKey) {
      return NextResponse.json(
        { error: "Pinecone API key isn't set" },
        { status: 401 }
      );
    }

    if (!pineconeIndexName) {
      return NextResponse.json(
        { error: "Pinecone Index Name is not set" },
        { status: 401 }
      );
    }

    const loader = new DocxLoader("public/documents/Support AI.docx");

    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 300,
      chunkOverlap: 20,
    });

    const documentChunks = await splitter.splitDocuments(docs);

    const pinecone = new Pinecone({
      apiKey: pineconeAPIKey,
    });

    const pineconeIndex = pinecone.Index(pineconeIndexName);

    const embedModel = new GoogleGenerativeAIEmbeddings({
      apiKey: geminiAPIKey,
      model: "embedding-001", // 768 dimensions
      taskType: TaskType.RETRIEVAL_DOCUMENT,
    });

    await PineconeStore.fromDocuments(documentChunks, embedModel, {
      pineconeIndex,
    });

    return NextResponse.json({ message: "Documents uploaded to Pinecone" });
  } catch (error: any) {
    return NextResponse.json({ message: error.message }, { status: 500 });
  }
};
