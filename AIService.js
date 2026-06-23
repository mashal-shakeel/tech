import { InferenceClient } from "@huggingface/inference";
import dotenv from "dotenv";

dotenv.config();

const client = new InferenceClient(process.env.HF_TOKEN);

async function runPrompt(
  prompt,
  temperature,
  maxTokens
) {
  try {
    const response = await client.chatCompletion({
      model: "meta-llama/Llama-3.1-8B-Instruct",

      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],

      temperature,
      max_tokens: maxTokens,
    });

    return response.choices[0].message.content;

  } catch (error) {
    console.error("HF Error:", error.message);
    throw error;
  }
}

export async function qualifyLead(leadData) {
  const prompt = `
You are a sales qualification expert.

Analyze this lead and return valid JSON only.

{
  "score": 0,
  "quality": "",
  "reason": "",
  "next_action": ""
}

Lead:
${JSON.stringify(leadData, null, 2)}
`;

  return runPrompt(prompt, 0.1, 300);
}


export async function classifyTicket(ticketText) {
  const prompt = `
You are a sales qualification expert. Classify the ticket and return valid JSON only.

{
  "department": "",
  "priority": "",
  "category": "",
  "summary": ""
}

Ticket:
${ticketText}
`;

  return runPrompt(prompt, 0.0, 300);
}

export async function draftEmail(context) {
  const prompt = `
You are a professional email writer.
Write a professional email. 

Context:
${context}

`;

  return runPrompt(prompt, 0.8, 600);
}

export async function extractData(rawText) {
  const prompt = `
You are a data extraction engine. Extract structured information. Return valid JSON only.

{
  "name": "",
  "email": "",
  "phone": "",
  "company": "",
  "other_details": []
}

Text:
${rawText}
`;

  return runPrompt(prompt, 0.0, 300);
}

async function main() {
  try {
    console.log("\n===== LEAD QUALIFICATION =====\n");

    const lead = await qualifyLead({
      company: "InvoZone",
      employees: 300,
      budget: "PKR50,000",
      industry: "Manufacturing",
      interest: "CRM System",
    });

    console.log(lead);

    console.log("\n===== TICKET CLASSIFICATION =====\n");

    const ticket = await classifyTicket(
      "Our payment gateway is failing and customers cannot checkout."
    );

    console.log(ticket);

    console.log("\n===== EMAIL DRAFTER =====\n");

    const email = await draftEmail(
      "Follow up after a demo call and schedule next meeting."
    );

    console.log(email);

    console.log("\n===== DATA EXTRACTION =====\n");

    const extracted = await extractData(`
      Mashal Shakeel
      mashal.shakeel@invozone.dev
      +92 300 1234567
      InvoZone
      Technical Trainee for AI and ML
    `);

    console.log(extracted);

  } catch (error) {
    console.error("Application Error:", error.message);
  }
}
