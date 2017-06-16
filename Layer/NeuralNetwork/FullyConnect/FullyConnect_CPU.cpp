//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"FullyConnect_DATA.hpp"
#include"FullyConnect_FUNC.hpp"
#include"FullyConnect_Base.h"

#include"FullyConnect_CPU.h"
#include"FullyConnect_LayerData_CPU.h"

#include"Library/NeuralNetwork/Optimizer.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	FullyConnect_CPU::FullyConnect_CPU(Gravisbell::GUID guid, FullyConnect_LayerData_CPU& i_layerData)
		:	FullyConnect_Base	(guid)
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	neuronCount						(0)		/**< �j���[������ */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
	{
	}
	/** �f�X�g���N�^ */
	FullyConnect_CPU::~FullyConnect_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 FullyConnect_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode FullyConnect_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	FullyConnect_LayerData_Base& FullyConnect_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const FullyConnect_LayerData_Base& FullyConnect_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_CPU::PreProcessLearn(U32 batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDOutputBuffer.resize(batchSize);

		// �o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDInputBuffer.resize(batchSize);

		// �p�����[�^�̕ω��ʃo�b�t�@
		this->lpDBias.resize(this->neuronCount);
		this->lpDNeuron.resize(this->neuronCount * this->inputBufferCount);


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_CPU::PreProcessCalculate(U32 batchSize)
	{
		this->batchSize = batchSize;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// �j���[���������m�F
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �j���[�����o�b�t�@�̃T�C�Y�m�F
		if(this->layerData.lpNeuron.size() != this->neuronCount * this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// ���̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->m_lppInputBuffer.resize(batchSize, NULL);

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum*this->outputBufferCount];
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();
		this->pLearnData->WriteToStruct((BYTE*)&this->learnData);

		switch(this->learnData.Optimizer)
		{
		case FullyConnect::LearnDataStructure::Optimizer_SGD:
			UpdateOptimizer_SGD_CPU(&this->m_pOptimizer_neuron, this->neuronCount*this->inputBufferCount, this->learnData.LearnCoeff);
			UpdateOptimizer_SGD_CPU(&this->m_pOptimizer_bias,   this->neuronCount,                        this->learnData.LearnCoeff);
			break;
		case FullyConnect::LearnDataStructure::Optimizer_Momentum:
			UpdateOptimizer_Momentum_CPU(&this->m_pOptimizer_neuron, this->neuronCount*this->inputBufferCount, this->learnData.LearnCoeff, this->learnData.Momentum_alpha);
			UpdateOptimizer_Momentum_CPU(&this->m_pOptimizer_bias,   this->neuronCount,                        this->learnData.LearnCoeff, this->learnData.Momentum_alpha);
			break;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_CPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FullyConnect_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				float tmp = 0;

				// �j���[�����̒l�����Z
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					tmp += this->m_lppInputBuffer[batchNum][inputNum] * this->layerData.lppNeuron[neuronNum][inputNum];
				}
				tmp += this->layerData.lpBias[neuronNum];

				// ������
				this->lppBatchOutputBuffer[batchNum][neuronNum] = tmp;

#ifdef _DEBUG
				if(isnan(this->lppBatchOutputBuffer[batchNum][neuronNum]))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
			}
		}


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode FullyConnect_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		memcpy(o_lpOutputBuffer, &this->lpOutputBuffer[0], sizeof(F32)*outputBufferCount * this->batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FullyConnect_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];

		
		// ���͌덷�������v�Z
		this->m_lpDInputBuffer = o_lppDInputBuffer;
		if(this->m_lpDInputBuffer)
		{
			// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				this->m_lppDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];

			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					float tmp = 0.0f;

					for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						tmp += this->m_lppDOutputBuffer[batchNum][neuronNum] * this->layerData.lppNeuron[neuronNum][inputNum];
					}

					this->m_lppDInputBuffer[batchNum][inputNum] = tmp;
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FullyConnect_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// �p�����[�^�ω��ʂ̃o�b�t�@���N���A
		memset(&this->lpDBias[0],   0, sizeof(F32)*this->lpDBias.size());
		memset(&this->lpDNeuron[0], 0, sizeof(F32)*this->lpDNeuron.size());

		// �w�K�덷���v�Z
		for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// �o�C�A�X�X�V
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					 sumDOutput += this->m_lppDOutputBuffer[batchNum][neuronNum];
				}

				this->lpDBias[neuronNum] = sumDOutput;
			}

			// ���͑Ή��j���[�����X�V
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					sumDOutput += this->m_lppInputBuffer[batchNum][inputNum] * this->m_lppDOutputBuffer[batchNum][neuronNum];
				}

				this->lpDNeuron[neuronNum*this->inputBufferCount + inputNum] += sumDOutput;
			}
		}

		// �덷�𔽉f
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->UpdateParameter(&this->layerData.lpBias[0],   &this->lpDBias[0]);
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->UpdateParameter(&this->layerData.lpNeuron[0], &this->lpDNeuron[0]);


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_CPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer;
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode FullyConnect_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		memcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32) * inputBufferCount * batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
