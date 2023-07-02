# Create a new weaviate vector database and store the result of the generated embeddings

import weaviate

def store_embeddings(embeddings):
    client = weaviate.Client(
        url="https://dev-docs-qa-bot-72hr7pq4.weaviate.network",
        auth_client_secret=weaviate.AuthApiKey(
            api_key="7Ue7892SsmmcncqN2CZIHOgB5lkTclJhbgGY"
        ),
    )

    # Uncomment if you want to improve update the data stored
    # client.schema.delete_class("PDFQA")

    schema = {
        "classes": [
            {
                "class": "PDFQA",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": False,
                        "vectorizeClassName": False,
                        "vectorizePropertyName": False,
                    }
                },
                "vectorizer": "text2vec-openai",
                "properties": [
                    {
                        "name": "embeddings",
                        "dataType": ["number"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False,
                                "vectorizeClassName": False,
                            }
                        },
                    }
                ],
            }
        ]
    }

    client.schema.create(schema)
    print('Schema created...')

    client.batch.configure(
        batch_size=10,
        dynamic=True,
        timeout_retries=3,
    )

    for i in range(0, len(embeddings)):
        item = embeddings.iloc[i]

        pdf_obj = {
            "embedded_values": item["embedded_values"],
        }

        try:
            client.batch.add_data_object(pdf_obj, "PDF")
        except BaseException as error:
            print("Import Failed at: ", i)
            print("An exception occurred: {}".format(error))
            # Stop the import on error
            break

        print("Status: ", str(i) + "/" + str(len(embeddings) - 1))

    client.batch.flush()
    print("Job done...")
    return True


