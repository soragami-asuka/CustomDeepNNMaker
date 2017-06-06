// IODataLayer.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include "Library/Layer/IOData/IODataLayer.h"


#include<vector>
#include<list>
#include<algorithm>

// UUID�֘A�p
#include<boost/uuid/uuid_generators.hpp>

namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerCPU : public IIODataLayer
	{
	private:
		Gravisbell::GUID guid;	/**< ����ID */
		Gravisbell::IODataStruct ioDataStruct;	/**< �f�[�^�\�� */

		std::vector<F32*> lpBufferList;

		U32 batchSize;	/**< �o�b�`�����T�C�Y */
		const U32* lpBatchDataNoList;	/**< �o�b�`�����f�[�^�ԍ����X�g */

		std::vector<F32> lpOutputBuffer;	/**< �o�̓o�b�t�@ */
		std::vector<F32> lpDInputBuffer;	/**< ���͌덷�o�b�t�@ */

		std::vector<F32*> lpBatchDataPointer;			/**< �o�b�`�����f�[�^�̔z��擪�A�h���X���X�g */
		std::vector<F32*> lpBatchDInputBufferPointer;	/**< �o�b�`�������͌덷�����̔z��擱�A�h���X���X�g */

		U32 calcErrorCount;	/**< �덷�v�Z�����s������ */
		std::vector<F32> lpErrorValue_min;	/**< �ŏ��덷 */
		std::vector<F32> lpErrorValue_max;	/**< �ő�덷 */
		std::vector<F32> lpErrorValue_ave;	/**< ���ό덷 */
		std::vector<F32> lpErrorValue_ave2;	/**< ���ϓ��덷 */
		std::vector<F32> lpErrorValue_crossEntropy;	/**< �N���X�G���g���s�[ */

		std::vector<S32> lpMaxErrorDataNo;

	public:
		/** �R���X�g���N�^ */
		IODataLayerCPU(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
			:	guid				(guid)
			,	ioDataStruct		(ioDataStruct)
			,	lpBatchDataNoList	(NULL)
			,	calcErrorCount		(0)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~IODataLayerCPU()
		{
			this->ClearData();
		}


		//===========================
		// ������
		//===========================
	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void)
		{
			return ErrorCode::ERROR_CODE_NONE;
		}


		//==============================
		// ���C���[���ʌn
		//==============================
	public:
		/** ���C���[��ʂ̎擾 */
		U32 GetLayerKind()const
		{
			return ELayerKind::LAYER_KIND_CPU | ELayerKind::LAYER_KIND_SINGLE_INPUT | ELayerKind::LAYER_KIND_SINGLE_OUTPUT | ELayerKind::LAYER_KIND_DATA;
		}

		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const
		{
			return this->guid;
		}

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const
		{
			Gravisbell::GUID layerCode;
			Gravisbell::Layer::IOData::GetLayerCode(layerCode);

			return layerCode;
		}

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const
		{
			return NULL;
		}


		//==============================
		// �f�[�^�Ǘ��n
		//==============================
	public:
		/** �f�[�^�̍\�������擾���� */
		IODataStruct GetDataStruct()const
		{
			return this->ioDataStruct;
		}

		/** �f�[�^�̃o�b�t�@�T�C�Y���擾����.
			@return �f�[�^�̃o�b�t�@�T�C�Y.�g�p����F32�^�z��̗v�f��. */
		U32 GetBufferCount()const
		{
			return this->ioDataStruct.GetDataCount();
		}

		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return	�ǉ����ꂽ�ۂ̃f�[�^�Ǘ��ԍ�. ���s�����ꍇ�͕��̒l. */
		Gravisbell::ErrorCode AddData(const F32 lpData[])
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			// �o�b�t�@�m��
			F32* lpBuffer = new F32[this->GetBufferCount()];
			if(lpBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_ALLOCATION_MEMORY;

			// �R�s�[
			memcpy(lpBuffer, lpData, sizeof(F32)*this->GetBufferCount());

			// ���X�g�ɒǉ�
			lpBufferList.push_back(lpBuffer);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �f�[�^�����擾���� */
		U32 GetDataCount()const
		{
			return (U32)this->lpBufferList.size();
		}
		/** �f�[�^��ԍ��w��Ŏ擾����.
			@param num		�擾����ԍ�
			@param o_lpBufferList �f�[�^�̊i�[��z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return ���������ꍇ0 */
		Gravisbell::ErrorCode GetDataByNum(U32 num, F32 o_lpBufferList[])const
		{
			if(num >= this->lpBufferList.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			if(o_lpBufferList == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			for(U32 i=0; i<this->GetBufferCount(); i++)
			{
				o_lpBufferList[i] = this->lpBufferList[num][i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
		/** �f�[�^��ԍ��w��ŏ������� */
		Gravisbell::ErrorCode EraseDataByNum(U32 num)
		{
			if(num >= this->lpBufferList.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// �ԍ��̏ꏊ�܂ňړ�
			auto it = this->lpBufferList.begin();
			for(U32 i=0; i<num; i++)
				it++;

			// �폜
			if(*it != NULL)
				delete *it;
			this->lpBufferList.erase(it);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �f�[�^��S��������.
			@return	���������ꍇ0 */
		Gravisbell::ErrorCode ClearData()
		{
			for(U32 i=0; i<lpBufferList.size(); i++)
			{
				if(lpBufferList[i] != NULL)
					delete lpBufferList[i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �o�b�`�����f�[�^�ԍ����X�g��ݒ肷��.
			�ݒ肳�ꂽ�l������GetDInputBuffer(),GetOutputBuffer()�̖߂�l�����肷��.
			@param i_lpBatchDataNoList	�ݒ肷��f�[�^�ԍ����X�g. [GetBatchSize()�̖߂�l]�̗v�f�����K�v */
		Gravisbell::ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[])
		{
			this->lpBatchDataNoList = i_lpBatchDataNoList;

			// �f�[�^���o�͗p�o�b�t�@�ɃR�s�[
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
					return Gravisbell::ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

				memcpy(this->lpBatchDataPointer[batchNum], this->lpBufferList[this->lpBatchDataNoList[batchNum]], sizeof(F32)*this->GetBufferCount());
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}



		//==============================
		// ���C���[���ʌn
		//==============================
	public:
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessLearn(U32 batchSize)
		{
			// �ʏ�̉��Z�p�̏��������s
			ErrorCode err = PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			// �덷�����f�[�^�z��̏�����
			this->lpDInputBuffer.resize(batchSize * this->GetBufferCount());
			this->lpBatchDInputBufferPointer.resize(batchSize);
			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				this->lpBatchDInputBufferPointer[batchNum] = &this->lpDInputBuffer[batchNum * this->GetBufferCount()];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessCalculate(U32 batchSize)
		{
			// �o�b�`�T�C�Y�̕ۑ�
			this->batchSize = batchSize;

			// �o�b�t�@�̊m�ۂƃo�b�`�����f�[�^�z��̏�����
			this->lpOutputBuffer.resize(batchSize * this->GetBufferCount());
			this->lpBatchDataPointer.resize(batchSize);
			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				this->lpBatchDataPointer[batchNum] = &this->lpOutputBuffer[batchNum * this->GetBufferCount()];
			}

			// �덷�v�Z����
			this->lpErrorValue_min.resize(this->GetBufferCount());
			this->lpErrorValue_max.resize(this->GetBufferCount());
			this->lpErrorValue_ave.resize(this->GetBufferCount());
			this->lpErrorValue_ave2.resize(this->GetBufferCount());
			this->lpErrorValue_crossEntropy.resize(this->GetBufferCount());

			this->lpMaxErrorDataNo.resize(this->GetBufferCount());

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& config)
		{
			return this->PreProcessCalculateLoop();
		}
		/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessCalculateLoop()
		{
			this->calcErrorCount = 0;
			this->lpErrorValue_min.assign(this->lpErrorValue_min.size(),  FLT_MAX);
			this->lpErrorValue_max.assign(this->lpErrorValue_max.size(),  0.0f);
			this->lpErrorValue_ave.assign(this->lpErrorValue_ave.size(),  0.0f);
			this->lpErrorValue_ave2.assign(this->lpErrorValue_ave2.size(), 0.0f);
			this->lpErrorValue_crossEntropy.assign(this->lpErrorValue_crossEntropy.size(), 0.0f);

			this->lpMaxErrorDataNo.assign(this->lpMaxErrorDataNo.size(), -1);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		U32 GetBatchSize()const
		{
			return this->batchSize;
		}


		//==============================
		// ���͌n
		//==============================
	public:
		/** �w�K�덷���v�Z����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v */
		Gravisbell::ErrorCode CalculateLearnError(Gravisbell::CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
		{
			U32 inputBufferCount = this->GetInputBufferCount();

			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 inputNum=0; inputNum<inputBufferCount; inputNum++)
				{
					F32 output = i_lppInputBuffer[batchNum*inputBufferCount + inputNum];
					F32 teach  = this->lpBatchDataPointer[batchNum][inputNum];

					F32 error = teach - output;
					F32 error_abs = abs(error);

					if(this->lpDInputBuffer.size() > 0)
					{
						this->lpBatchDInputBufferPointer[batchNum][inputNum] = error;
//						this->lpDInputBuffer[batchNum][inputNum] = -(output - teach) / (output * (1.0f - output));
					}

					if(this->lpErrorValue_max[inputNum] < error_abs)
						this->lpMaxErrorDataNo[inputNum] = this->lpBatchDataNoList[batchNum];

					F32 crossEntropy = -(F32)(
						      teach  * log(max(0.0001,  output)) +
						 (1 - teach) * log(max(0.0001,1-output))
						 );

					// �덷��ۑ�
					this->lpErrorValue_min[inputNum]  = min(this->lpErrorValue_min[inputNum], error_abs);
					this->lpErrorValue_max[inputNum]  = max(this->lpErrorValue_max[inputNum], error_abs);
					this->lpErrorValue_ave[inputNum]  += error_abs;
					this->lpErrorValue_ave2[inputNum] += error_abs * error_abs;
					this->lpErrorValue_crossEntropy[inputNum] += crossEntropy;
				}
				this->calcErrorCount++;
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}


		/** �덷�̒l���擾����.
			CalculateLearnError()��1��ȏ���s���Ă��Ȃ��ꍇ�A����ɓ��삵�Ȃ�.
			@param	o_min	�ŏ��덷.
			@param	o_max	�ő�덷.
			@param	o_ave	���ό덷.
			@param	o_ave2	���ϓ��덷. */
		ErrorCode GetCalculateErrorValue(F32& o_min, F32& o_max, F32& o_ave, F32& o_ave2, F32& o_crossEntropy)
		{
			o_min  = FLT_MAX;
			o_max  = 0.0f;
			o_ave  = 0.0f;
			o_ave2 = 0.0f;
			o_crossEntropy = 0.0f;

			for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
			{
				o_min   = min(o_min, this->lpErrorValue_min[inputNum]);
				o_max   = max(o_max, this->lpErrorValue_max[inputNum]);
				o_ave  += this->lpErrorValue_ave[inputNum];
				o_ave2 += this->lpErrorValue_ave2[inputNum];
				o_crossEntropy += this->lpErrorValue_crossEntropy[inputNum];
			}

			o_ave  = o_ave / this->calcErrorCount / this->GetBufferCount();
			o_ave2 = (F32)sqrt(o_ave2 / this->calcErrorCount / this->GetBufferCount());
			o_crossEntropy = o_crossEntropy / this->calcErrorCount / this->GetBufferCount();

			//for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
			//	printf("%d,", this->lpMaxErrorDataNo[inputNum]);
			//printf("\n");

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** �ڍׂȌ덷�̒l���擾����.
			�e���o�͂̒l���Ɍ덷�����.
			CalculateLearnError()��1��ȏ���s���Ă��Ȃ��ꍇ�A����ɓ��삵�Ȃ�.
			�e�z��̗v�f����[GetBufferCount()]�ȏ�ł���K�v������.
			@param	o_lpMin		�ŏ��덷.
			@param	o_lpMax		�ő�덷.
			@param	o_lpAve		���ό덷.
			@param	o_lpAve2	���ϓ��덷. */
		ErrorCode GetCalculateErrorValueDetail(F32 o_lpMin[], F32 o_lpMax[], F32 o_lpAve[], F32 o_lpAve2[])
		{
			for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
			{
				o_lpMin[inputNum]   = this->lpErrorValue_min[inputNum];
				o_lpMax[inputNum]   = this->lpErrorValue_max[inputNum];
				o_lpAve[inputNum]  += this->lpErrorValue_ave[inputNum] / this->GetDataCount();
				o_lpAve2[inputNum] += (F32)sqrt(this->lpErrorValue_ave2[inputNum] / this->GetDataCount());
			}
			
			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const
		{
			return this->GetDataStruct();
		}

		/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		U32 GetInputBufferCount()const
		{
			return this->GetBufferCount();
		}

		/** �w�K�������擾����.
			�z��̗v�f����GetInputBufferCount�̖߂�l.
			@return	�덷�����z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const
		{
			return &this->lpDInputBuffer[0];
		}
		/** �w�K�������擾����.
			@param lpDOutputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
		Gravisbell::ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
		{
			if(o_lpDInputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 inputBufferCount = this->GetOutputBufferCount();

			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(&o_lpDInputBuffer[batchNum*inputBufferCount], this->lpBatchDataPointer[batchNum], sizeof(F32)*inputBufferCount);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}


		//==============================
		// �o�͌n
		//==============================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const
		{
			return this->GetDataStruct();
		}

		/** �o�̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		U32 GetOutputBufferCount()const
		{
			return this->GetBufferCount();
		}

		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpOutputBuffer[0];
		}
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		Gravisbell::ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(&o_lpOutputBuffer[batchNum*outputBufferCount], this->lpBatchDataPointer[batchNum], sizeof(F32)*outputBufferCount);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPU(Gravisbell::IODataStruct ioDataStruct)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataLayerCPUwithGUID(uuid.data, ioDataStruct);
	}
	/** ���͐M���f�[�^���C���[���쐬����.CPU����
		@param guid			���C���[��GUID.
		@param bufferSize	�o�b�t�@�̃T�C�Y.��F32�^�z��̗v�f��.
		@return	���͐M���f�[�^���C���[�̃A�h���X */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPUwithGUID(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataLayerCPU(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell


