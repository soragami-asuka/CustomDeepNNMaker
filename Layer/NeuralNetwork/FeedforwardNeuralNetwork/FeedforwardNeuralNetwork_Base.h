//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_BASE_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_BASE_H__

#include<Layer/NeuralNetwork/INeuralNetwork.h>
#include<Layer/NeuralNetwork/ILayerDLLManager.h>


#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>

#include"LayerConnect.h"
#include"LayerConnectInput.h"
#include"LayerConnectOutput.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	class FeedforwardNeuralNetwork_Base : public INeuralNetwork
	{
	private:
		// �f�[�^�{��
		class FeedforwardNeuralNetwork_LayerData_Base& layerData;

		std::map<Gravisbell::GUID, ILayerConnect*>	lpLayerInfo;	/**< �S���C���[�̊Ǘ��N���X. <���C���[GUID, ���C���[�ڑ����̃A�h���X> */
		std::vector<ILayerData*>	lpTemporaryLayerData;			/**< �ꎞ�ۑ����ꂽ���C���[�f�[�^. */

		std::list<ILayerConnect*> lpCalculateLayerList;		/**< ���C���[���������ɕ��ׂ����X�g.  */

		const Gravisbell::GUID guid;			/**< ���C���[���ʗp��GUID */
		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */
		IODataStruct outputDataStruct;	/**< �o�̓f�[�^�\�� */

		SettingData::Standard::IData* pLearnData;		/**< �w�K�ݒ���`�����R���t�B�O�N���X */

		U32 batchSize;	/**< �o�b�`�T�C�Y */

	protected:
		LayerConnectInput  inputLayer;	/**< ���͐M���̑�փ��C���[�̃A�h���X. */
		LayerConnectOutput outputLayer;	/**< �o�͐M���̑�փ��C���[�̃A�h���X. */

		Gravisbell::Common::ITemporaryMemoryManager* pLocalTemporaryMemoryManager;
		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

		// ���o�̓o�b�t�@
		std::vector<F32>		lpInputBuffer;		/**< ���̓o�b�t�@ <�o�b�`��><���͐M����> */

		CONST_BATCH_BUFFER_POINTER	m_lppInputBuffer;	/**< �O������a���������̓o�b�t�@�̃A�h���X(���Z�f�o�C�X�ˑ�) */
		BATCH_BUFFER_POINTER	m_lppDInputBuffer;	/**< �O������a���������͌덷�o�b�t�@�̃A�h���X(���Z�f�o�C�X�ˑ�) */
		BATCH_BUFFER_POINTER	m_lppDOutputBuffer;	/**< �O������a�������o�͌덷�o�b�t�@�̃A�h���X(���Z�f�o�C�X�ˑ�) */

	public:
		//====================================
		// �R���X�g���N�^/�f�X�g���N�^
		//====================================
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_Base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct, Gravisbell::Common::ITemporaryMemoryManager* i_pTemporaryMemoryManager);
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_Base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** �f�X�g���N�^ */
		virtual ~FeedforwardNeuralNetwork_Base();


		//====================================
		// ���C���[�̒ǉ�
		//====================================
	public:
		/** ���C���[��ǉ�����.
			�ǉ��������C���[�̏��L����NeuralNetwork�Ɉڂ邽�߁A�������̊J�������Ȃǂ͑S��INeuralNetwork���ōs����.
			@param pLayer	�ǉ����郌�C���[�̃A�h���X. */
		ErrorCode AddLayer(ILayerBase* pLayer);

		/** �ꎞ���C���[��ǉ�����.
			�ǉ��������C���[�f�[�^�̏��L����NeuralNetwork�Ɉڂ邽�߁A�������̊J�������Ȃǂ͑S��INeuralNetwork���ōs����.
			@param	i_pLayerData	�ǉ����郌�C���[�f�[�^.
			@param	o_player		�ǉ����ꂽ���C���[�̃A�h���X. */
		virtual ErrorCode AddTemporaryLayer(ILayerData* i_pLayerData, ILayerBase** o_pLayer, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** ���C���[���폜����.
			@param i_guid	�폜���郌�C���[��GUID */
		ErrorCode EraseLayer(const Gravisbell::GUID& i_guid);
		/** ���C���[��S�폜���� */
		ErrorCode EraseAllLayer();

		/** �o�^����Ă��郌�C���[�����擾���� */
		U32 GetLayerCount()const;
		/** ���C���[��GUID��ԍ��w��Ŏ擾���� */
		ErrorCode GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid);


		//====================================
		// ���o�̓��C���[
		//====================================
	public:
		/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
		GUID GetInputGUID()const;

		/** �o�͐M�����C���[��ݒ肷�� */
		ErrorCode SetOutputLayerGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// ���C���[�̐ڑ�
		//====================================
	public:
		/** ���C���[�ɓ��̓��C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);
		/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID);
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID);

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		ErrorCode GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		ErrorCode GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;


		/** ���C���[�̐ڑ���ԂɈُ킪�Ȃ����`�F�b�N����.
			@param	o_errorLayer	�G���[�������������C���[GUID�i�[��. 
			@return	�ڑ��Ɉُ킪�Ȃ��ꍇ��NO_ERROR, �ُ킪�������ꍇ�ُ͈���e��Ԃ��A�Ώۃ��C���[��GUID��o_errorLayer�Ɋi�[����. */
		ErrorCode CheckAllConnection(Gravisbell::GUID& o_errorLayer);

		//====================================
		// ���͌덷�o�b�t�@�֘A
		//====================================
	private:
		/** �e���C���[���g�p������͌덷�o�b�t�@�����蓖�Ă� */
		ErrorCode AllocateDInputBuffer(void);

	protected:
		/** ���͌덷�o�b�t�@�̑�����ݒ肷�� */
		virtual ErrorCode SetDInputBufferCount(U32 i_DInputBufferCount) = 0;

		/** ���͌덷�o�b�t�@�̃T�C�Y��ݒ肷�� */
		virtual ErrorCode ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize) = 0;

	public:
		/** ���͌덷�o�b�t�@���擾����(�����f�o�C�X�ˑ�) */
		virtual BATCH_BUFFER_POINTER GetDInputBuffer_d(U32 i_DINputBufferNo) = 0;


		//====================================
		// �o�̓o�b�t�@�֘A
		//====================================
	private:
		/** �e���C���[���g�p����o�̓o�b�t�@�����蓖�Ă� */
		ErrorCode AllocateOutputBuffer(void);

	protected:
		/** �o�̓o�b�t�@�̑�����ݒ肷�� */
		virtual ErrorCode SetOutputBufferCount(U32 i_outputBufferCount) = 0;

		/** �o�̓o�b�t�@�̃T�C�Y��ݒ肷�� */
		virtual ErrorCode ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize) = 0;

	public:
		/** �o�̓o�b�t�@�̌��݂̎g�p�҂��擾���� */
		virtual GUID GetReservedOutputBufferID(U32 i_i_outputBufferNo) = 0;
		/** �o�̓o�b�t�@���g�p���ɂ��Ď擾����(�����f�o�C�X�ˑ�) */
		virtual BATCH_BUFFER_POINTER ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid) = 0;


		//====================================
		// �O������a���������o�̓o�b�t�@�֘A
		//====================================
	public:
		/** ���̓o�b�t�@���擾����(�����f�o�C�X�ˑ�) */
		CONST_BATCH_BUFFER_POINTER GetInputBuffer();
		/** ���̓o�b�t�@���擾����(�����f�o�C�X�ˑ�) */
		CONST_BATCH_BUFFER_POINTER GetInputBuffer_d();
		/** ���͌덷�o�b�t�@���擾����(�����f�o�C�X�ˑ�) */
		BATCH_BUFFER_POINTER GetDInputBuffer_d();
		/** �o�͌덷�o�b�t�@���擾����(�����f�o�C�X�ˑ�) */
		BATCH_BUFFER_POINTER GetDOutputBuffer_d();


		//====================================
		// �w�K�ݒ�
		//====================================
	public:
		/** ���s���ݒ���擾����. */
		const SettingData::Standard::IData* GetRuntimeParameter()const;
		SettingData::Standard::IData* GetRuntimeParameter();

		/** �w�K�ݒ���擾����.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			@param	guid	�擾�Ώۃ��C���[��GUID. */
		const SettingData::Standard::IData* GetRuntimeParameter(const Gravisbell::GUID& guid)const;
		SettingData::Standard::IData* GetRuntimeParameter(const Gravisbell::GUID& guid);

		/** �w�K�ݒ�̃A�C�e�����擾����.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID. */
		SettingData::Standard::IItemBase* GetRuntimeParameterItem(const Gravisbell::GUID& guid, const wchar_t* i_dataID);

		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param);
		ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param);
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			int�^�Afloat�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param);
		ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param);
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			bool�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param);
		ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param);
		/** �w�K�ݒ��ݒ肷��.
			�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
			string�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param);
		ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param);


		//====================================
		// ���o�̓o�b�t�@�֘A
		//====================================
	public:
		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		virtual ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;


		//===========================
		// ���C���[����
		//===========================
	public:
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKindBase(void)const;

		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[�̎�ގ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		U32 GetBatchSize()const;

		/** �ꎞ�o�b�t�@�Ǘ��N���X���擾���� */
		Common::ITemporaryMemoryManager& GetTemporaryMemoryManager();

		//================================
		// ����������
		//================================
	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const;

		/** ���̓o�b�t�@�����擾����. */
		U32 GetInputBufferCount()const;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;
		IODataStruct GetOutputDataStruct(const GUID& i_layerGUID)const;

		/** �o�̓o�b�t�@�����擾���� */
		U32 GetOutputBufferCount()const;


		//================================
		// ���Z����
		//================================
	protected:
		/** �ڑ��̊m�����s�� */
		ErrorCode EstablishmentConnection(void);

	public:
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessLearn(U32 batchSize);
		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessCalculate(unsigned int batchSize);
		
		/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessLoop();


		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);


		//================================
		// �w�K����
		//================================
	public:
		/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

		/** �w�K���������s����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif	// __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_H__