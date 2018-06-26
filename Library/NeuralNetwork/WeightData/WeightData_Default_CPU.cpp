//=====================================
// �d�݃f�[�^�N���X.CPU����
// �f�t�H���g.
//=====================================
#include"stdafx.h"

#include<vector>

#include"WeightData_Default.h"

#include<Layer/NeuralNetwork/IOptimizer.h>
#include<Library/NeuralNetwork/Optimizer.h>
#include<Library/NeuralNetwork/Initializer.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class WeightData_Default_CPU : public IWeightData
	{
	private:
		std::vector<F32> lpWeight;
		std::vector<F32> lpBias;

		IOptimizer* m_pOptimizer_weight;	/**< �d�ݍX�V�p�I�v�e�B�}�C�U */
		IOptimizer* m_pOptimizer_bias;		/**< �o�C�A�X�X�V�p�I�v�e�B�}�C�U */


	public:
		//===========================
		// �R���X�g���N�^/�f�X�g���N�^
		//===========================
		/** �R���X�g���N�^ */
		WeightData_Default_CPU(U32 i_neuronCount, U32 i_inputCount)
			:	lpWeight			(i_neuronCount * i_inputCount)
			,	lpBias				(i_neuronCount)
			,	m_pOptimizer_weight	(NULL)
			,	m_pOptimizer_bias	(NULL)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~WeightData_Default_CPU()
		{
			if(this->m_pOptimizer_weight)
				delete this->m_pOptimizer_weight;
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
		}

	public:
		//===========================
		// ������
		//===========================
		ErrorCode Initialize(const wchar_t i_initializerID[], U32 i_inputCount, U32 i_outputCount)
		{
			auto& initializer = Gravisbell::Layer::NeuralNetwork::GetInitializerManager().GetInitializer(i_initializerID);

			// �j���[����
			for(unsigned int weightNum=0; weightNum<this->lpWeight.size(); weightNum++)
			{
				lpWeight[weightNum] = initializer.GetParameter(i_inputCount, i_outputCount);
			}
			// �o�C�A�X
			for(unsigned int biasNum=0; biasNum<this->lpBias.size(); biasNum++)
			{
				this->lpBias[biasNum] = initializer.GetParameter(i_inputCount, i_outputCount);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
		S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize)
		{
			S64 readBufferByte = 0;
			
			// �j���[�����W��
			memcpy(&this->lpWeight[0], &i_lpBuffer[readBufferByte], this->lpWeight.size() * sizeof(F32));
			readBufferByte += (int)this->lpWeight.size() * sizeof(F32);

			// �o�C�A�X
			memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(F32));
			readBufferByte += (int)this->lpBias.size() * sizeof(F32);


			// �I�v�e�B�}�C�U
			S64 useBufferSize = 0;
			// weight
			if(this->m_pOptimizer_weight)
				delete this->m_pOptimizer_weight;
			this->m_pOptimizer_weight = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// bias
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
			this->m_pOptimizer_bias = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;

			return readBufferByte;
		}


		//===========================
		// �T�C�Y���擾
		//===========================
		/** Weight�̃T�C�Y���擾���� */
		U64 GetWeigthSize()const
		{
			return this->lpWeight.size();
		}
		/** Bias�̃T�C�Y���擾���� */
		U64 GetBiasSize()const
		{
			return this->lpBias.size();
		}


		//===========================
		// �l���擾
		//===========================
		/** Weight���擾���� */
		const F32* GetWeight()const
		{
			return &this->lpWeight[0];
		}
		/** Bias���擾���� */
		const F32* GetBias()const
		{
			return &this->lpBias[0];
		}


		//===========================
		// �l���X�V
		//===========================
		/** Weigth,Bias��ݒ肷��.
			@param	lpWeight	�ݒ肷��Weight�̒l.
			@param	lpBias		�ݒ肷��Bias�̒l. */
		ErrorCode SetData(const F32* i_lpWeight, const F32* i_lpBias)
		{
			memcpy(&this->lpWeight[0], i_lpWeight, sizeof(F32)*this->lpWeight.size());
			memcpy(&this->lpBias[0],   i_lpBias,   sizeof(F32)*this->lpBias.size());

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** Weight,Bias���X�V����.
			@param	lpDWeight	Weight�̕ω���.
			@param	lpDBias		Bias��h�ω���. */
		ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias)
		{
			// �덷�𔽉f
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->UpdateParameter(&this->lpWeight[0], i_lpDWeight);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->UpdateParameter(&this->lpBias[0],   i_lpDBias);

			return ErrorCode::ERROR_CODE_NONE;
		}


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
		/** �I�v�e�B�}�C�U�[��ύX���� */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[])
		{
			ChangeOptimizer_CPU(&this->m_pOptimizer_bias,   i_optimizerID, (U32)this->lpBias.size());
			ChangeOptimizer_CPU(&this->m_pOptimizer_weight, i_optimizerID, (U32)this->lpWeight.size());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
		{
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->SetHyperParameter(i_parameterID, i_value);

			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
		{
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->SetHyperParameter(i_parameterID, i_value);
		
			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
		{
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->SetHyperParameter(i_parameterID, i_value);

			return ErrorCode::ERROR_CODE_NONE;
		}
		
		//===========================
		// ���C���[�ۑ�
		//===========================
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U64 GetUseBufferByteCount()const
		{
			U64 bufferSize = 0;

			// �{�̂̃o�C�g��
			bufferSize += sizeof(F32) * this->lpWeight.size();	// �d�݌W��
			bufferSize += sizeof(F32) * this->lpBias.size();	// �o�C�A�X�W��

			// �I�v�e�B�}�C�U�[�̃o�C�g��
			bufferSize += this->m_pOptimizer_weight->GetUseBufferByteCount();
			bufferSize += this->m_pOptimizer_bias->GetUseBufferByteCount();

			return bufferSize;
		}
		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			S64 writeBufferByte = 0;

			// �j���[�����W��
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpWeight[0], this->lpWeight.size() * sizeof(F32));
			writeBufferByte += (int)this->lpWeight.size() * sizeof(F32);
			// �o�C�A�X
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], this->lpBias.size() * sizeof(F32));
			writeBufferByte += (int)this->lpBias.size() * sizeof(F32);

			// �I�v�e�B�}�C�U
			// weight
			writeBufferByte += this->m_pOptimizer_weight->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			// bias
			writeBufferByte += this->m_pOptimizer_bias->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

			return writeBufferByte;
		}
	};

	/** �d�݃N���X���쐬����.
		�f�t�H���g.CPU����. */
	IWeightData* CreateWeightData_Default_CPU(U32 i_neuronCount, U32 i_inputCount)
	{
		return new WeightData_Default_CPU(i_neuronCount, i_inputCount);
	}
}
}
}