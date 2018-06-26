//=====================================
// �d�݃f�[�^�N���X.CPU����
// �f�t�H���g.
//=====================================
#include"stdafx.h"

#include<vector>

#include"WeightData_WeightNormalization.h"

#include<Layer/NeuralNetwork/IOptimizer.h>
#include<Library/NeuralNetwork/Optimizer.h>
#include<Library/NeuralNetwork/Initializer.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class WeightData_WeightNormalization_CPU : public IWeightData
	{
	private:
		std::vector<F32> lpWeight;
		std::vector<F32> lpBias;

		std::vector<F32> lpScale;			/**< neuron */
		std::vector<F32> lpVector;			/**< neuron*input */
		std::vector<F32> lpVectorScale;		/**< vector�̑傫�� neuron */

		// �덷�p
		std::vector<F32> lpDScale;
		std::vector<F32> lpDVector;

		IOptimizer* m_pOptimizer_scale;		/**< �X�J���[�̍X�V�p�I�v�e�B�}�C�U */
		IOptimizer* m_pOptimizer_vector;	/**< �x�N�^�[�̍X�V�p�I�v�e�B�}�C�U */
		IOptimizer* m_pOptimizer_bias;		/**< �o�C�A�X�X�V�p�I�v�e�B�}�C�U */

		U32 neuronCount;
		U32 inputCount;

	public:
		//===========================
		// �R���X�g���N�^/�f�X�g���N�^
		//===========================
		/** �R���X�g���N�^ */
		WeightData_WeightNormalization_CPU(U32 i_neuronCount, U32 i_inputCount)
			:	lpWeight			(i_neuronCount * i_inputCount)
			,	lpBias				(i_neuronCount)
			
			,	lpScale				(i_neuronCount)
			,	lpVector			(i_neuronCount * i_inputCount)
			,	lpVectorScale		(i_neuronCount)

			,	lpDScale			(i_neuronCount)
			,	lpDVector			(i_neuronCount * i_inputCount)

			,	m_pOptimizer_scale	(NULL)
			,	m_pOptimizer_vector	(NULL)
			,	m_pOptimizer_bias	(NULL)

			,	neuronCount			(i_neuronCount)
			,	inputCount			(i_inputCount)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~WeightData_WeightNormalization_CPU()
		{
			if(this->m_pOptimizer_scale)
				delete this->m_pOptimizer_scale;
			if(this->m_pOptimizer_vector)
				delete this->m_pOptimizer_vector;
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

			// �d��
			std::vector<F32> lpTmpWeight(this->lpWeight.size());
			for(unsigned int weightNum=0; weightNum<lpTmpWeight.size(); weightNum++)
			{
				lpTmpWeight[weightNum] = initializer.GetParameter(i_inputCount, i_outputCount);
			}
			// �o�C�A�X
			std::vector<F32> lpTmpBias(this->lpBias.size());
			for(unsigned int biasNum=0; biasNum<lpTmpBias.size(); biasNum++)
			{
//				lpTmpBias[biasNum] = initializer.GetParameter(i_inputCount, i_outputCount);
				lpTmpBias[biasNum] = 0.0f;
			}

			return this->SetData(&lpTmpWeight[0], &lpTmpBias[0]);
		}
		S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize)
		{
			S64 readBufferByte = 0;
			
			// �X�P�[��
			memcpy(&this->lpScale[0], &i_lpBuffer[readBufferByte], this->lpScale.size() * sizeof(F32));
			readBufferByte += (int)this->lpScale.size() * sizeof(F32);
			
			// �x�N�^�[
			memcpy(&this->lpVector[0], &i_lpBuffer[readBufferByte], this->lpVector.size() * sizeof(F32));
			readBufferByte += (int)this->lpVector.size() * sizeof(F32);

			// �o�C�A�X
			memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(F32));
			readBufferByte += (int)this->lpBias.size() * sizeof(F32);


			// �I�v�e�B�}�C�U
			S64 useBufferSize = 0;
			// scale
			if(this->m_pOptimizer_scale)
				delete this->m_pOptimizer_scale;
			this->m_pOptimizer_scale = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// vector
			if(this->m_pOptimizer_vector)
				delete this->m_pOptimizer_vector;
			this->m_pOptimizer_vector = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// bias
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
			this->m_pOptimizer_bias = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;

			// �x�N�^�[�̃X�P�[�����Čv�Z
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// �X�P�[���Z�o
				F32 sumValue = 0.0f;
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					F32 value = this->lpVector[neuronNum*this->inputCount + inputNum];

					sumValue += (value * value);
				}
				this->lpVectorScale[neuronNum] = sqrtf(sumValue);
			}

			// �d�݂��X�V
			this->UpdateWeight();

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
			// Bias���R�s�[
			memcpy(&this->lpBias[0],   i_lpBias,   sizeof(F32)*this->lpBias.size());

			// �X�P�[�����Z�o���āA�x�N�^�[�̃T�C�Y��1�ɂ���
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// �X�P�[���Z�o
				F32 sumValue = 0.0f;
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					F32 value = i_lpWeight[neuronNum*this->inputCount + inputNum];

					sumValue += (value * value);
				}
				F32 scale = sqrtf(sumValue);
				this->lpScale[neuronNum] = scale;

				// �x�N�^�[�T�C�Y��1�ɂ���
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					this->lpVector[neuronNum*this->inputCount + inputNum] = i_lpWeight[neuronNum*this->inputCount + inputNum] / scale;
				}
				this->lpVectorScale[neuronNum] = 1.0f;
			}

			// �d�݂��Čv�Z
			this->UpdateWeight();

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** Weight,Bias���X�V����.
			@param	lpDWeight	Weight�̕ω���.
			@param	lpDBias		Bias��h�ω���. */
		ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias)
		{
			// �덷���v�Z
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				F32 vectorScale = this->lpVectorScale[neuronNum];

				// �X�P�[���덷
				F32 sumValue = 0.0f;
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					U32 offset = neuronNum*this->inputCount + inputNum;

					sumValue += this->lpVector[offset] * i_lpDWeight[offset];
				}
				this->lpDScale[neuronNum] = sumValue / vectorScale;

				// �x�N�g���덷
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					U32 offset = neuronNum*this->inputCount + inputNum;

					this->lpDVector[offset] = (this->lpScale[neuronNum] / vectorScale) * (i_lpDWeight[offset] - this->lpDScale[neuronNum]*this->lpVector[offset]/vectorScale);
				}
			}


			// �덷�𔽉f
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->UpdateParameter(&this->lpScale[0], &this->lpDScale[0]);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->UpdateParameter(&this->lpVector[0], &this->lpDVector[0]);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->UpdateParameter(&this->lpBias[0],   i_lpDBias);

			// �X�P�[�����Čv�Z
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// �X�P�[���Z�o
				F32 sumValue = 0.0f;
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					F32 value = this->lpVector[neuronNum*this->inputCount + inputNum];

					sumValue += (value * value);
				}
				this->lpVectorScale[neuronNum] = sqrtf(sumValue);
			}

			// �d�݂��Čv�Z
			this->UpdateWeight();

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** Weight���X�V */
		void UpdateWeight()
		{
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					this->lpWeight[neuronNum*this->inputCount + inputNum] = this->lpScale[neuronNum] * this->lpVector[neuronNum*this->inputCount + inputNum] / this->lpVectorScale[neuronNum];
				}
			}
		}

		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
		/** �I�v�e�B�}�C�U�[��ύX���� */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[])
		{
			ChangeOptimizer_CPU(&this->m_pOptimizer_scale,  i_optimizerID, (U32)this->lpScale.size());
			ChangeOptimizer_CPU(&this->m_pOptimizer_vector, i_optimizerID, (U32)this->lpVector.size());
			ChangeOptimizer_CPU(&this->m_pOptimizer_bias,   i_optimizerID, (U32)this->lpBias.size());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);

			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		
			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);

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
			bufferSize += sizeof(F32) * this->lpScale.size();	// �X�P�[��
			bufferSize += sizeof(F32) * this->lpVector.size();	// �x�N�^�[
			bufferSize += sizeof(F32) * this->lpBias.size();	// �o�C�A�X�W��

			// �I�v�e�B�}�C�U�[�̃o�C�g��
			bufferSize += this->m_pOptimizer_scale->GetUseBufferByteCount();
			bufferSize += this->m_pOptimizer_vector->GetUseBufferByteCount();
			bufferSize += this->m_pOptimizer_bias->GetUseBufferByteCount();

			return bufferSize;
		}
		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			S64 writeBufferByte = 0;

			// �X�P�[��
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpScale[0], this->lpScale.size() * sizeof(F32));
			writeBufferByte += (int)this->lpScale.size() * sizeof(F32);
			// �x�N�^�[
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpVector[0], this->lpVector.size() * sizeof(F32));
			writeBufferByte += (int)this->lpVector.size() * sizeof(F32);
			// �o�C�A�X
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], this->lpBias.size() * sizeof(F32));
			writeBufferByte += (int)this->lpBias.size() * sizeof(F32);

			// �I�v�e�B�}�C�U
			// scale
			writeBufferByte += this->m_pOptimizer_scale->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			// vector
			writeBufferByte += this->m_pOptimizer_vector->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			// bias
			writeBufferByte += this->m_pOptimizer_bias->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

			return writeBufferByte;
		}
	};

	/** �d�݃N���X���쐬����.
		�f�t�H���g.CPU����. */
	IWeightData* CreateWeightData_WeightNormalization_CPU(U32 i_neuronCount, U32 i_inputCount)
	{
		return new WeightData_WeightNormalization_CPU(i_neuronCount, i_inputCount);
	}
}
}
}