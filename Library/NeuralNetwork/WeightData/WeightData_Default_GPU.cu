//=====================================
// �d�݃f�[�^�N���X.GPU����
// �f�t�H���g.
//=====================================
#include<thrust/device_vector.h>

#include"WeightData_Default.h"

#include<Layer/NeuralNetwork/IOptimizer.h>
#include<Library/NeuralNetwork/Optimizer.h>
#include<Library/NeuralNetwork/Initializer.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class WeightData_Default_GPU : public IWeightData
	{
	private:
		thrust::device_vector<F32> lpWeight;
		thrust::device_vector<F32> lpBias;

		IOptimizer* m_pOptimizer_weight;	/**< �d�ݍX�V�p�I�v�e�B�}�C�U */
		IOptimizer* m_pOptimizer_bias;		/**< �o�C�A�X�X�V�p�I�v�e�B�}�C�U */


	public:
		//===========================
		// �R���X�g���N�^/�f�X�g���N�^
		//===========================
		/** �R���X�g���N�^ */
		WeightData_Default_GPU(U32 i_neuronCount, U32 i_inputCount)
			:	lpWeight			(i_neuronCount * i_inputCount)
			,	lpBias				(i_neuronCount)
			,	m_pOptimizer_weight	(NULL)
			,	m_pOptimizer_bias	(NULL)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~WeightData_Default_GPU()
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

			thrust::host_vector<F32> lpTmpWeight(this->lpWeight.size());
			thrust::host_vector<F32> lpTmpBias(this->lpBias.size());

			for(U32 i=0; i<lpTmpWeight.size(); i++)
			{
				lpTmpWeight[i] = initializer.GetParameter(i_inputCount, i_outputCount);
			}
			for(U32 i=0; i<lpTmpBias.size(); i++)
			{
				lpTmpBias[i] = initializer.GetParameter(i_inputCount, i_outputCount);
			}

			this->lpWeight = lpTmpWeight;
			this->lpBias   = lpTmpBias;

			return ErrorCode::ERROR_CODE_NONE;
		}
		S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize)
		{
			S64 readBufferByte = 0;
			
			// �o�b�t�@����R�s�[
			// �j���[����
			cudaMemcpy(
				thrust::raw_pointer_cast(&this->lpWeight[0]),
				&i_lpBuffer[readBufferByte],
				sizeof(F32) * this->lpWeight.size(),
				cudaMemcpyHostToDevice);
			readBufferByte += sizeof(F32) * this->lpWeight.size();

			// �o�C�A�X
			cudaMemcpy(
				thrust::raw_pointer_cast(&this->lpBias[0]),
				&i_lpBuffer[readBufferByte],
				sizeof(F32) * this->lpBias.size(),
				cudaMemcpyHostToDevice);
			readBufferByte += sizeof(F32) * this->lpBias.size();


			// �I�v�e�B�}�C�U
			S64 useBufferSize = 0;
			// neuron
			if(this->m_pOptimizer_weight)
				delete this->m_pOptimizer_weight;
			this->m_pOptimizer_weight = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// bias
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
			this->m_pOptimizer_bias = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
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
			return thrust::raw_pointer_cast(&this->lpWeight[0]);
		}
		/** Bias���擾���� */
		const F32* GetBias()const
		{
			return thrust::raw_pointer_cast(&this->lpBias[0]);
		}


		//===========================
		// �l���X�V
		//===========================
		/** Weigth,Bias��ݒ肷��.
			@param	lpWeight	�ݒ肷��Weight�̒l.
			@param	lpBias		�ݒ肷��Bias�̒l. */
		ErrorCode SetData(const F32* i_lpWeight, const F32* i_lpBias)
		{
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpWeight[0]), i_lpWeight, sizeof(F32)*this->lpWeight.size(), cudaMemcpyDeviceToDevice);
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpBias[0]),   i_lpBias,   sizeof(F32)*this->lpBias.size(), cudaMemcpyDeviceToDevice);

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** Weight,Bias���X�V����.
			@param	lpDWeight	Weight�̕ω���.
			@param	lpDBias		Bias��h�ω���. */
		ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias)
		{
			// �덷�𔽉f
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->UpdateParameter(thrust::raw_pointer_cast(&this->lpWeight[0]), i_lpDWeight);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->lpBias[0]),   i_lpDBias);

			return ErrorCode::ERROR_CODE_NONE;
		}


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
		/** �I�v�e�B�}�C�U�[��ύX���� */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[])
		{
			ChangeOptimizer_GPU(&this->m_pOptimizer_bias,   i_optimizerID, (U32)this->lpBias.size());
			ChangeOptimizer_GPU(&this->m_pOptimizer_weight, i_optimizerID, (U32)this->lpWeight.size());

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
			cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpWeight[0]), this->lpWeight.size() * sizeof(F32), cudaMemcpyDeviceToHost);
			writeBufferByte += (int)this->lpWeight.size() * sizeof(F32);
			// �o�C�A�X
			cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpBias[0]), this->lpBias.size() * sizeof(F32), cudaMemcpyDeviceToHost);
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
	IWeightData* CreateWeightData_Default_GPU(U32 i_neuronCount, U32 i_inputCount)
	{
		return new WeightData_Default_GPU(i_neuronCount, i_inputCount);
	}
}
}
}