//===================================
// ���o�̓f�[�^���Ǘ�����N���X
// GPU����
//===================================
#include "Library/Layer/IOData/IODataLayer.h"


#include<vector>
#include<list>
#include<algorithm>

// UUID�֘A�p
#include<boost/uuid/uuid_generators.hpp>

// CUDA�p
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerGPU_base : public IIODataLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< ����ID */
		Gravisbell::IODataStruct ioDataStruct;	/**< �f�[�^�\�� */


		U32 batchSize;	/**< �o�b�`�����T�C�Y */
		const U32* lpBatchDataNoList;	/**< �o�b�`�����f�[�^�ԍ����X�g */


		thrust::device_vector<F32>	lpOutputBuffer;	/**< �o�̓o�b�t�@ */
		thrust::device_vector<F32>	lpDInputBuffer;	/**< ���͌덷�o�b�t�@ */

		U32 calcErrorCount;	/**< �덷�v�Z�����s������ */
		thrust::device_vector<F32>	lpErrorValue_max;	/**< �ő�덷 */
		thrust::device_vector<F32>	lpErrorValue_ave;	/**< ���ό덷 */
		thrust::device_vector<F32>	lpErrorValue_ave2;	/**< ���ϓ��덷 */
		thrust::device_vector<F32>	lpErrorValue_crossEntropy;	/**< �N���X�G���g���s�[ */

		cublasHandle_t cublasHandle;

	public:
		/** �R���X�g���N�^ */
		IODataLayerGPU_base(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);
		/** �f�X�g���N�^ */
		virtual ~IODataLayerGPU_base();


		//===========================
		// ������
		//===========================
	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);


		//==============================
		// ���C���[���ʌn
		//==============================
	public:
		/** ���C���[��ʂ̎擾 */
		U32 GetLayerKind()const;

		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;

		//==============================
		// �f�[�^�Ǘ��n
		//==============================
	public:
		/** �f�[�^�̍\�������擾���� */
		IODataStruct GetDataStruct()const;

		/** �f�[�^�̃o�b�t�@�T�C�Y���擾����.
			@return �f�[�^�̃o�b�t�@�T�C�Y.�g�p����F32�^�z��̗v�f��. */
		U32 GetBufferCount()const;

		/** �f�[�^��ǉ�����.
			@param	lpData	�f�[�^��g�̔z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return	�ǉ����ꂽ�ۂ̃f�[�^�Ǘ��ԍ�. ���s�����ꍇ�͕��̒l. */
		virtual Gravisbell::ErrorCode AddData(const F32 lpData[]) = 0;

		/** �f�[�^�����擾���� */
		virtual U32 GetDataCount()const = 0;

		/** �f�[�^��ԍ��w��Ŏ擾����.
			@param num		�擾����ԍ�
			@param o_lpBufferList �f�[�^�̊i�[��z��. GetBufferSize()�̖߂�l�̗v�f�����K�v.
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode GetDataByNum(U32 num, F32 o_lpBufferList[])const = 0;

		/** �f�[�^��ԍ��w��ŏ������� */
		virtual Gravisbell::ErrorCode EraseDataByNum(U32 num) = 0;

		/** �f�[�^��S��������.
			@return	���������ꍇ0 */
		virtual Gravisbell::ErrorCode ClearData() = 0;

		/** �o�b�`�����f�[�^�ԍ����X�g��ݒ肷��.
			�ݒ肳�ꂽ�l������GetDInputBuffer(),GetOutputBuffer()�̖߂�l�����肷��.
			@param i_lpBatchDataNoList	�ݒ肷��f�[�^�ԍ����X�g. [GetBatchSize()�̖߂�l]�̗v�f�����K�v */
		virtual Gravisbell::ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[]) = 0;



		//==============================
		// ���C���[���ʌn
		//==============================
	public:
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessLearn(U32 batchSize);

		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessCalculate(U32 batchSize);

		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& config);
		/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		Gravisbell::ErrorCode PreProcessCalculateLoop();

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		U32 GetBatchSize()const;


		//==============================
		// ���͌n
		//==============================
	public:
		/** �w�K�덷���v�Z����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v */
		Gravisbell::ErrorCode CalculateLearnError(Gravisbell::CONST_BATCH_BUFFER_POINTER i_lppInputBuffer);


		/** �덷�̒l���擾����.
			CalculateLearnError()��1��ȏ���s���Ă��Ȃ��ꍇ�A����ɓ��삵�Ȃ�.
			@param	o_min	�ŏ��덷.
			@param	o_max	�ő�덷.
			@param	o_ave	���ό덷.
			@param	o_ave2	���ϓ��덷. */
		Gravisbell::ErrorCode GetCalculateErrorValue(F32& o_max, F32& o_ave, F32& o_ave2, F32& o_crossEntropy);

		/** �ڍׂȌ덷�̒l���擾����.
			�e���o�͂̒l���Ɍ덷�����.
			CalculateLearnError()��1��ȏ���s���Ă��Ȃ��ꍇ�A����ɓ��삵�Ȃ�.
			�e�z��̗v�f����[GetBufferCount()]�ȏ�ł���K�v������.
			@param	o_lpMin		�ŏ��덷.
			@param	o_lpMax		�ő�덷.
			@param	o_lpAve		���ό덷.
			@param	o_lpAve2	���ϓ��덷. */
		Gravisbell::ErrorCode GetCalculateErrorValueDetail(F32 o_lpMax[], F32 o_lpAve[], F32 o_lpAve2[]);


	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const;

		/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		U32 GetInputBufferCount()const;

		/** �w�K�������擾����.
			�z��̗v�f����GetInputBufferCount�̖߂�l.
			@return	�덷�����z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const;
		/** �w�K�������擾����.
			@param lpDOutputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
		Gravisbell::ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const;


		//==============================
		// �o�͌n
		//==============================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;

		/** �o�̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		U32 GetOutputBufferCount()const;

		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		Gravisbell::ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;
	};


}	// IOData
}	// Layer
}	// Gravisbell
