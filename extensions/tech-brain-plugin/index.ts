import axios from 'axios';

/**
 * OpenClaw Tech-Brain Bridge Plugin
 * This connects the persistent agent to the local 1,920-dimension vector matrix.
 */

export const activate = (api: any) => {
  const RAG_ENDPOINT = 'http://localhost:8000/search';

  // Register the Technical Knowledge Tool
  api.registerTool({
    name: "query_technical_knowledge",
    description: "Accesses the internal Knowledge Base for technical manuals, OWS logs, and Alphanumeric Site IDs. Use this for deep technical verification.",
    schema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The technical query or specific ID (e.g., OWS workflow, RCA alarm code)."
        },
        deep_search: {
          type: "boolean",
          description: "Enable agentic deep research if the first search is insufficient.",
          default: false
        }
      },
      required: ["query"]
    },
    async execute({ query, deep_search }: { query: string, deep_search?: boolean }) {
      try {
        console.log(`[TechBrain] Agent is querying: ${query}`);
        
        // Route to either basic search or agentic researcher
        const endpoint = deep_search ? 'http://localhost:8000/agentic_search' : RAG_ENDPOINT;
        
        const response = await axios.post(endpoint, {
          query: query,
          limit: 5
        });

        const data = response.data;
        
        // Format the response for the agent
        let output = `### KNOWLEDGE RETRIEVAL RESULTS\n\n`;
        output += `**Synthesis:** ${data.answer}\n\n`;
        
        if (data.confidence) {
          output += `**Confidence:** ${data.confidence} (${data.confidence_score}/5)\n`;
        }

        if (data.sources && data.sources.length > 0) {
          output += `\n**Top Sources Used:**\n`;
          data.sources.forEach((src: any, i: number) => {
            const filename = src.metadata?.filename || 'Unknown Document';
            output += `${i + 1}. ${filename} (Score: ${src.score.toFixed(2)})\n`;
          });
        }

        return output;

      } catch (error: any) {
        console.error(`[TechBrain] Error during RAG retrieval:`, error.message);
        return `Failed to reach the Technical Knowledge Base. Ensure the RAG FastAPI service is running at ${RAG_ENDPOINT}.`;
      }
    }
  });

  console.log("✅ Technical Knowledge Bridge successfully activated in OpenClaw.");
};
