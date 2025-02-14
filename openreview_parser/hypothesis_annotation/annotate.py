"""Utility functions for annotating OpenReview submissions with Hypothesis annotations."""
import os
import logging

import backoff
import openai
import tiktoken

from langchain.schema.messages import BaseMessage
from langchain_openai import ChatOpenAI


from openreview_parser.utils.data import Paper

EXAMPLE_PAPER = """Introduction 
What is encoded in vector representations of textual data, and can we control it? Word embeddings, pre-trained language models, and more generally deep learning methods emerge as very effective techniques for text classification. Accordingly, they are increasingly being used for predictions in real-world situations. A large part of the success is due to the models' ability to perform representation learning, coming up with effective feature representations for the prediction task at hand. However, these learned representations, while effective, are also notoriously opaque: we do not know what is encoded in them. Indeed, there is an emerging line of work on probing deep-learning derived representations for syntactic (Linzen et al., 2016; Hewitt and Manning, 2019;, semantic and factual knowledge (Petroni et al., 2019). There is also evidence that they capture a lot of information regarding the demographics of the author of the text (Blodgett et al., 2016;Elazar and Goldberg, 2018) What can we do in situations where we do not want our representations to encode certain kinds of information? For example, we may want a word representation that does not take tense into account, or that does not encode part-of-speech distinctions. We may want a classifier that judges the formality of the text, but which is also oblivious to the topic the text was taken from. Finally, and also our empirical focus in this work, this situation often arises when considering fairness and bias of languagebased classification. We may not want our wordembeddings to encode gender stereotypes, and we do not want sensitive decisions on hiring or loan approvals to condition on the race, gender or age of the applicant. We present a novel method for selectively removing specific kinds of information from a representation. Previous methods are either based on projection on a pre-specified, user-provided direction (Bolukbasi et al., 2016), or on adding an adversarial objective to an end-to-end training process. Both of these have benefits and limitations, as we discuss in the related work section ( §2). Our proposed method, Iterative Nullspace Projection (INLP), presented in section 4, can be seen as a combination of these approaches, capitalizing on the benefits of both. Like the projection methods, it is also based on the mathematical notion of linear projection, a commonly used de-terministic operator. Like the adversarial methods, it is data-driven in the directions it removes: we do not presuppose specific directions in the latent space that correspond to the protected attribute, but rather learn those directions, and remove them. Empirically, we find it to work well. We evaluate the method on the challenging task of removing gender signals from word embeddings Zhao et al., 2018). Recently, Gonen and Goldberg (2019) showed several limitations of current methods for this task. We show that our method is effective in reducing many, but not all, of these ( §4).We also consider the context of fair classification, where we want to ensure that a classifier's decision is oblivious to a protected attribute such as race, gender or age. There, we need to integrate the projection-based method within a pre-trained classifier. We propose a method to do so in section §5, and demonstrate its effectiveness in a controlled setup ( §6.2) as well as in a real-world one ( §6.3). Finally, while we propose a general purpose information-removal method, our main evaluation is in the realm of bias and fairness applications. We stress that this calls for some stricter scrutiny, as the effects of blindly trusting strong claims can have severe real-world consequences on individuals. We discuss the limitations of our model in the context of such applications in section §7. \
Objective and Definitions 
Our main goal is to guard sensitive information, so that it will not be encoded in a representation. Given a set of vectors x i ∈ R d , and corresponding discrete attributes Z, z i ∈ {1, ..., k} (e.g. race or gender), we aim to learn a transformation g : R d → R d , such that z i cannot be predicted from g(x i ). In this work we are concerned with linear guarding: we seek a guard g such that no linear classifier w(•) can predict z i from g(x i ) with an accuracy greater than that of a decision rule that considers only the proportion of labels in Z. We also wish for g(x i ) to stay informative: when the vectors x are used for some end task, we want g(x) to have as minimal influence as possible on the end task performance, provided that z remains guarded. We use the following definitions: Guarded w.r.t. a hypothesis class Let X = x 1 , ..., x m ∈ X ⊆ R d be a set of vectors, with corresponding discrete attributes Z, z i ∈ {1, ..., k}. We say the set X is guarded for Z with respect to hypothesis class H (conversely Z is guarded in X) if there is no classifier W ∈ H that can predict z i from x i at better than guessing the majority class. Guarding function A function g : R n → R n is said to be guarding X for Z \
Iterative Nullspace Projection
Given a set of vectors x i ∈ R d and a set of corresponding discrete protected attributes z i ∈ Z, we seek a linear guarding function g that remove the linear dependence between Z and X. We begin with a high-level description of our approach. Let c be a trained linear classifier, parameterized by a matrix W ∈ R k×d , that predicts a property z with some accuracy. We can construct a projection matrix P such that W (P x) = 0 for all x, rendering W useless on dataset X . We then iteratively train additional classifiers W and perform the same procedure, until no more linear information regarding Z remains in X. Constructing P is achieved via nullspace projection, as described below. This method is the core of the INLP algorithm (Algorithm 1). Nullspace Projection The linear interaction between W and a new test point x has a simple geometric interpretation: x is projected on the subspace spanned by W 's rows, and is classified according to the dot product between x and W 's rows, which is proportional to the components of x in the direction of W 's rowpsace. Therefore, if we zeroed all components of x in the direction of W 's row-space, we removed all information used by W for prediction: the decision boundary found by the classifier is no longer useful. As the orthogonal component of the rowspace is the nullspace, zeroing those components of x is equivalent to projecting x on W 's nullspace. Figure illustrates the idea for the 2 dimensional binary-classification setting, in which W is just a 2-dimensional vector. For an algebraic interpretation, recall that the null-space of a matrix W is defined as the space N (W ) = {x|W x = 0}. Given the basis vectors of N (W ) we can construct a projection matrix P N (W ) into N (W ), yielding W (P N (W ) x) = 0 ∀x. This suggests a simple method for rendering z linearly guarded for a set of vectors X: training a linear classifier that is parameterized by W 0 to predict Z from X, calculating its nullspace, finding the orthogonal projection matrix P N (W 0 ) onto the nullspace, and using it to remove from X those components that were used by the classifier for predicting Z. Note that the orthogonal projection P N (w 0 ) is the least harming linear operation to remove the linear information captured by W 0 from X, in the sense that among all maximum rank (which is not full, as such transformations are invertible-hence not linearly guarding) projections onto the nullspace of W 0 , it carries the least impact on distances. This is so since the image under an orthogonal projection into a subspace is by definition the closest vector in that subspace. Iterative Projection Projecting the inputs X on the nullspace of a single linear classifier does not suffice for making Z linearly guarded: classifiers can often still be trained to recover z from the projected x with above chance accuracy, as there are often multiple linear directions (hyperplanes) that can partially capture a relation in multidimensional space. This can be remedied with an iterative process: After obtaining P N (W 0 ) , we train classifier W 1 on P N (W 0 ) X, obtain a projection matrix P N (W 1 ) , train a classifier W 2 on P N (W 1 ) P N (W 0 ) X and so on, until no classifier W m+1 can be trained. We return the guarding projection matrix P = P N (Wm) P N (W m-1 ) ...P N (W 0 ) , with the guarding function g(x) = P x. Crucially, the ith classifier W i is trained on the data X after the projection on the nullspaces of classifiers W 0 , ..., W i-1 and is therefore trained to find separating planes that are independent of the separating planes found by previous classifiers.In Appendix §A.1 we prove three desired proprieties of INLP: (1) any two protected-attribute classifiers found in INLP are orthogonal (Lemma A.1); (2) while in general the product of projection matrices is not a projection, the product P calculated in INLP is a valid projection (Corollary A.1.2); and (3) it projects any vector to the intersection of the nullspaces of each of the classifiers found in INLP, that is, after n INLP iterations, P is a projection to. We further bound the damage P causes to the structure of the space (Lemma A.2). INLP can thus be seen as a linear dimensionalityreduction method, which keeps only those directions in the latent space which are not indicative of the protected attribute.N (W 0 ) ∩ N (W 1 ) • • • ∩ N (W n ) (Corollary A During iterative nullspace projection, the property z becomes increasingly linearly-guarded in P x. For binary protected attributes, each intermediate W j is a vector, and the nullspace rank is d-1. Therefore, after n iterations, if the original rank of X was r, the rank of the projected input g(X) is at least r -n. The entire process is formalized in Algorithm 1. \
Application to Fair Classification 
The previous section described the INLP method for producing a linearly guarding function g for a set of vectors. We now turn to describe its usage in the context of providing fair classification by a (possibly deep) neural network classifier. In this setup, we are given, in addition to X and Z also labels Y , and wish to construct a classifier f : X → Y , while being fair with respect to Z. Fairness in classification can be defined in many ways. We focus on a notion of fairness by which the predictor f is oblivious to Z when making predictions about Y. To use linear guardedness in the context of a deep network, recall that a classification network f (x) can be decomposed into an encoder enc followed by a linear layer W :f (x) = W • enc(x),where W is the last layer of the network and enc is the rest of the network. If we can make sure that Z is linearly guarded in the inputs to W , then W will have no knowledge of Z when making its prediction about Y , making the decision process oblivious to Z. Adversarial training methods attempt to achieve such obliviousness by adding an adversarial objective to make enc(x) itself guarding. We take a different approach and add a guarding function on top of an already trained enc. We propose the following procedure. Given a training set X,Y and protected attribute Z, we first train a neural network f = W • enc(X) to best predict Y . This results in an encoder that extracts effective features from X for predicting Y . We then consider the vectors enc(X), and use the INLP method to produce a linear guarding function g that guards Z in enc(X). At this point, we can use the classifier W • g(enc(x)) to produce oblivious decisions, however by introducing g (which is lower rank than enc(x)) we may have harmed W s performance. We therefore freeze the network and fine-tune only W to predict Y from g(enc(x)), producing the final fair classifier f (x) = W • g(enc(x)). Notice that W only sees vectors which are linearly guarded for Z during its training, and therefore cannot take Z into ac-count when making its predictions, ensuring fair classification. We note that our notion of fairness by obliviousness does not, in the general case, correspond to other fairness metrics, such as equality of odds or of opportunity. It does, however, correlate with fairness metrics, as we demonstrate empirically. Further refinement. Guardedness is a property that holds in expectation over an entire dataset. For example, when considering a dataset of individuals from certain professions (as we do in §6.3), it is possible that the entire dataset is guarded for gender, yet if we consider only a subset of individuals (say, only nurses), we may still be able to recover gender with above majority accuracy, in that sub-population. As fairness metrics are often concerned with classification behavior also within groups, we propose the following refinement to the algorithm, which we use in the experiments in §6.2 and §6.3: in each iteration, we train a classifier to predict the protected attribute not on the entire training set, but only on the training examples belonging to a single (randomly chosen) main-task class (e.g. profession). By doing so, we push the protected attribute to be linearly guarded in the examples belonging to each of the main-task labels."
"""

EXAMPLE_HYPOTHESIS = "Problem: Neural representations in language-based classifiers often encode societal biases, compromising fairness without significantly affecting performance. Solution: Iterative Nullspace Projection (INLP) can remove specific types of information, such as societal biases, from these neural representations. By iteratively training linear classifiers to predict an attribute (e.g., gender), and projecting the representations onto the null space of these classifiers, INLP ensures that the attribute can no longer be predicted. This process reduces biases by eliminating linearly separable information, improving fairness without significantly affecting classifier performance."

MODELNAME2CONTEXT_SIZE = {"gpt-3.5-turbo": 16000}


def annotate_paper(paper: Paper, config: dict) -> str | None:
    """
    Annotate the given paper with its research hypothesis.

    Args:
        paper (Paper): The paper object to be annotated.
        config (dict): Configuration settings for the annotation process.

    Returns:
        str: The research hypothesis of the paper, or None if the context is too long.
    """
    logging.info(f"Annotating paper {paper.title} with its research hypothesis")

    encoder = tiktoken.encoding_for_model(config["model_name"])
    chat_model = get_chat_model(config)

    # Get prompts
    system_message = get_system_message()
    human_message = get_human_message(paper)

    # Check number of tokens
    system_message_token = encoder.encode(system_message, disallowed_special=())
    n_token_system_message = len(system_message_token)
    logging.info(f"System message #token: {n_token_system_message}")

    human_message_token = encoder.encode(human_message, disallowed_special=())
    n_token_human_message = len(human_message_token)
    logging.info(f"Human message #token {n_token_human_message}")

    if config["model_name"] not in MODELNAME2CONTEXT_SIZE:
        raise ValueError(
            f"For model name {config['model_name']} context size not found."
        )

    if (
        n_token_human_message + n_token_system_message
        > MODELNAME2CONTEXT_SIZE[config["model_name"]]
    ):
        logging.info(
            f"Paper {[paper.title]} context too long: {n_token_human_message+n_token_system_message}"
        )
        return None

    messages = [system_message, human_message]

    output = get_llm_response(chat_model, messages)
    print(output)
    if output is not None:
        hypothesis = str(output.content)
    else:
        hypothesis = None

    return hypothesis


def get_chat_model(config: dict) -> ChatOpenAI:
    """
    Get a chat model based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the chat model.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class.

    Raises:
        NotImplementedError: If the specified chat model is not implemented.
    """
    assert (
        os.environ.get("OPENAI_API_KEY", None) is not None
    ), "Please set OPENAI_API_KEY environment variable"
    if config["provider"] == "openai":
        model = ChatOpenAI(
            temperature=config["temperature"],
            model=config["model_name"],
            max_completion_tokens=config["max_tokens"],
        )
    else:
        raise NotImplementedError(f"Chat model: {config['provider']} not implemented")

    return model


def get_human_message(paper: Paper) -> str:
    """
    Get a the human prompt to annotate the given paper with its hypothesis.

    Args:
        paper (str): The text of the paper.

    Returns:
        str: Part of the prompt to annotate the paper with its hypothesis.
    """
    paper_text = paper.get_text(False)
    human_message = f"Annotate the following paper with its hypothesis: {paper_text}"
    return human_message


@backoff.on_exception(
    backoff.expo,
    openai.BadRequestError,
    max_tries=5,
    raise_on_giveup=False,
    on_giveup=lambda x: None,
)
def get_llm_response(chat_model: ChatOpenAI, messages: list[str]) -> BaseMessage:
    """
    Get the response from the chat model.

    Args:
        chat_model (ChatOpenAI): The chat model to use for generating the response.
        messages (list[str]): The list of messages in the conversation.

    Returns:
        AIMessage: The generated response from the chat model.
    """
    return chat_model.invoke(messages)


def get_system_message() -> str:
    """
    Return a system message with the task description, requirements, and an example of a paper with annotated hypothesis.

    Returns:
        str: System message with task description, requirements, and example.
    """
    return f"""
        Task Description:
        You are a PhD student tasked to annotate research papers with the hypothesis they investigate.
        You will be provided with infos about the paper.
        Your task is to extract the research hypothesis from this provided text.

        Requirements:
        - Clarity: Ensure the hypothesis is clearly stated and understandable without additional context.
        - Completeness: The hypothesis should be self-contained, including all necessary components such as the variables involved and the expected relationship or outcome, it should not contain information about potential results that come from an investigation of the hypothesis.
        - Terminology: Use precise and field-specific terminology that a research scientist in the relevant or adjacent field would understand.
        - Conciseness: Keep the hypothesis one to two sentences long, avoiding unnecessary details or jargon.

        Format:
        - Problem: The problem that the paper is addressing
        - Solution: The solution that the paper is proposing


        Here is one example of a paper and the annotated hypothesis:

        Paper:
        {EXAMPLE_PAPER}

        Annotated Hypothesis:
        {EXAMPLE_HYPOTHESIS}
        """
